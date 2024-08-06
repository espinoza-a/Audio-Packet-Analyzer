import numpy as np
import rtmixer
import math
import sounddevice as sd
import time
from scipy import signal
from scipy.ndimage import convolve
import copy
from datetime import datetime
import socket
import threading
import json


def get_input_devices() -> list:
    devices = sd.query_devices()

    input_devices = [device for device in devices if device['max_input_channels'] > 0]

    if len(input_devices) == 0:
        return []
    
    try:
        default_input_device = sd.query_devices(kind='input')
    except sd.PortAudioError:
        print("Failed to query the default input device")
        default_input_device = None
    
    input_devices = []
    if default_input_device is not None:
        default_input_device['index'] = devices.index(default_input_device)
        input_devices += [default_input_device]

    for device in devices:
        if device['max_input_channels'] > 0:
            device['index'] = devices.index(device)
            if default_input_device is not None and device['index'] != default_input_device['index']:
                input_devices += [device]

    return input_devices

class RingBuffer:
    FRAMES_PER_BUFFER = 512*4
    CHANNELS = 1
    DOWNSAMPLE = 1
    WINDOW = 10000

    def __init__(self, device):
        self.samplerate = device['default_samplerate']

        length = int(self.WINDOW * self.samplerate / (1000 * self.DOWNSAMPLE))
        self.data = np.ones((length, self.CHANNELS))

        self.stream = rtmixer.Recorder(device=device['index'], 
                                        channels=self.CHANNELS, 
                                        blocksize=self.FRAMES_PER_BUFFER,
                                        latency='low', 
                                        samplerate=self.samplerate)
        
        self.ringBufferSize = 2**int(math.log2(3 * self.samplerate))
        self.ringBuffer = rtmixer.RingBuffer(self.CHANNELS * self.stream.samplesize, self.ringBufferSize)

    def update_ring_buffer(self):
        self.ringBuffer = rtmixer.RingBuffer(self.CHANNELS * self.stream.samplesize, self.ringBufferSize)
        self.stream.record_ringbuffer(self.ringBuffer)

    def get_data(self):
        while self.ringBuffer.read_available >= self.FRAMES_PER_BUFFER:
            read, buf1, _ = self.ringBuffer.get_read_buffers(self.FRAMES_PER_BUFFER)
            assert read == self.FRAMES_PER_BUFFER
            buffer = np.frombuffer(buf1, dtype='float32')
            buffer.shape = -1, self.CHANNELS
            buffer = buffer[::self.DOWNSAMPLE]

            assert buffer.base.base == buf1
            shift = len(buffer)
            self.data = np.roll(self.data, -shift, axis=0)
            self.data[-shift:, :] = buffer
            self.ringBuffer.advance_read_index(self.FRAMES_PER_BUFFER)

        return self.data
    
    def get_stream(self):
        return self.stream


class KissListener:
    LISTENER_HOST = "127.0.0.1"
    LISTENER_PORT = 6025
    LISTENER_TIMEOUT = 90

    def __init__(self, dnlink_mode : set ):
        self.dnlink_mode = dnlink_mode
        self.sock = socket.socket()
        self.sock.settimeout(self.LISTENER_TIMEOUT)
        self.mutex = threading.Lock()
        self.is_kiss_connected = False

        try:
            self.sock.bind((self.LISTENER_HOST, self.LISTENER_PORT))
        except socket.error as e:
            print(str(e))

        print(f'Server is listing on the port {self.LISTENER_PORT}...')
        self.sock.listen()

        try:  
            Client, address = self.sock.accept()
            print('Connected to: ' + address[0] + ':' + str(address[1]))
            self.is_kiss_connected = True
            threading.Thread( target=self.client_handler, args=(Client, ) ).start()
        except:
            print('No connection from KISS')

    def client_handler(self, connection):
        while True:
            data = connection.recv(2048)
            message = data.decode('utf-8')
            if message == 'CLOSE':
                 break
            
            if message != None and message != '':
                msg = json.loads(message)
                if 'mode' in msg.keys():
                    self.handle_set_mode_msg(msg, datetime.now())
    
        connection.close()      

    def handle_set_mode_msg(self, msg, current_time):
        self.mutex.acquire(blocking=True)        
        if msg['mode'] == 'uplink':
            self.dnlink_mode["mode"] = ( current_time, False )
        if msg['mode'] == 'dnlink':
            self.dnlink_mode["mode"] = ( current_time, True )  
        self.mutex.release()


class AudioFilter:
    FILTER_WINDOW = 1024 * 8
    FILTER_THRESHOLD = 0.65

    def __init__(self, ring_buffer : RingBuffer):
        self.beg = datetime.now()

        self.ring_buffer : RingBuffer = ring_buffer
        self.samplerate = ring_buffer.samplerate

        self.stack = []
        self.dnlink_info : dict = { "mode" : (self.beg, False) }
        self.last_dnlink : dict = { "mode" : (self.beg, False) }
        self.tcp_server = KissListener(self.dnlink_info)
        self.is_kiss_connected = self.tcp_server.is_kiss_connected

        if self.is_kiss_connected:
            self.perform_data(self.ring_buffer.get_data())

    def perform_data(self, data : np.ndarray):
        f, _, Sxx = self.__filter__(data[-self.FILTER_WINDOW * 2:,:], self.samplerate)
        freq_arr = self.__find_signals__(f, Sxx)
        self.__resolve_data__(freq_arr)
        pass

    def __resolve_data__(self, freq_arr : np.ndarray):
        if self.__is_dnlink__() and not self.__is_mode_swiched__():
            if freq_arr.shape[0] > 0:
                self.stack += freq_arr.tolist()
        elif self.__is_dnlink__() and self.__is_mode_swiched__():
            self.__update_mode__()
        elif not self.__is_dnlink__() and self.__is_mode_swiched__():
            if len(self.stack) > 0:
                self.__log_data__(self.stack)
            self.stack.clear()
            self.__update_mode__()
        elif not self.__is_dnlink__() and not self.__is_mode_swiched__():
            pass 

    def __is_dnlink__(self):
        return self.dnlink_info["mode"][1]

    def __is_mode_swiched__(self):
        return self.dnlink_info["mode"][1] != self.last_dnlink["mode"][1]
    
    def __update_mode__(self):
        self.last_dnlink["mode"] = self.dnlink_info["mode"]

    def __log_data__(self, stack):
        print(self.last_dnlink["mode"][0], sum(stack) / len(stack))
    
    def __filter__(self, data : np.ndarray, samplerate : float):
        Ndft = self.FILTER_WINDOW
        Ndft_over = Ndft
        noverl = round(Ndft*0.8)
        window = signal.windows.hamming(Ndft)

        f, t, Sxx = signal.spectrogram(
            data[:,0], 
            fs=samplerate, 
            window=window, 
            noverlap=noverl, 
            nfft=Ndft_over,
            scaling='spectrum',
            mode='magnitude'
        )

        mag = copy.deepcopy(Sxx.real)
        row_count = np.shape(mag)[0]

        tmp = mag[(f > 500) & (f < 2.5e3), :]
        rms = np.sqrt(np.mean( tmp**2, axis=0))

        sel = mag > np.ones((row_count, 1)) * rms * 1.3
        mag[~sel] = 0

        # p_lo_cut = 0.4
        # sel2 = np.sum(mag, axis=0) / row_count > p_lo_cut
        # mag[~sel] = 0

        mag[f<30,:] = 0
        window = self.__create_window__(f[1])
        mag = self.__conv2d__(mag, window) 
        mag[(f <= 500) | (f >= 2.5e3), :] = 0

        return f, t, mag
    
    def __conv2d__(self, x, y):
        if (len(x.shape) < len(y.shape)):
            dim = x.shape
            for i in range(len(x.shape),len(y.shape)):
                dim = (1,) + dim
            x = x.reshape(dim)
        elif (len(y.shape) < len(x.shape)):
            dim = y.shape
            for i in range(len(y.shape),len(x.shape)):
                dim = (1,) + dim
            y = y.reshape(dim)

        origin = ()
        for i in range(len(x.shape)):
            if ( (x.shape[i] - y.shape[i]) % 2 == 0 and
                x.shape[i] > 1 and
                y.shape[i] > 1):
                origin = origin + (-1,)
            else:
                origin = origin + (0,)

        z = convolve(x,y, mode='constant', origin=origin)

        return z
    
    def __create_window__(self, f):
        f_w, f_s = 30, 600 # Hz

        n_w = int(np.ceil(f_w / f / 2) * 2)
        n_s = int(np.ceil(f_s / f))
        s_w = (np.ones((n_w, 1)), np.zeros((n_s - n_w, 1)), np.ones((n_w, 1)), np.zeros((n_s - n_w, 1)), np.ones((n_w, 1)))
        s_w = np.concatenate(s_w, axis=0)

        return s_w * np.ones((1,3))
    
    def __find_signals__(self, f, mag_conv) -> np.ndarray:
        M, I = mag_conv.max(axis=0), mag_conv.argmax(axis=0)
        Freq = f[I]

        M_cut    = M[M>self.FILTER_THRESHOLD]
        Freq_cut = Freq[M>self.FILTER_THRESHOLD]

        ind_cut = signal.find_peaks(M_cut)[0]

        if ind_cut.shape[0] > 0:
            Freq_cut = Freq_cut[ind_cut]

        return Freq_cut


if __name__ == '__main__':
    audio = get_input_devices()[0] # todo: refactor
    ring_buffer = RingBuffer(audio)
    audio_filter = AudioFilter(ring_buffer)

    with ring_buffer.get_stream():
        ring_buffer.update_ring_buffer()
        while True:
            data = ring_buffer.get_data() 
            audio_filter.perform_data(data)
            time.sleep(0.01)





