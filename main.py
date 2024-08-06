from PyQt5 import QtWidgets
from threading import Thread
from PyQt5.uic import loadUi
from mplwidget import MatplotlibWidget

import sys
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import time
from scipy import signal

from audio_filter import get_input_devices, RingBuffer, AudioFilter

class MyWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(MyWindow, self).__init__()
        loadUi('main_window.ui', self)
        self.spectrum_widget = MatplotlibWidget(self.mpl_widget)
        self.fig = self.spectrum_widget.figure
        self.gridLayout.addWidget(self.spectrum_widget, 0, 0, 1, 1) # put it on self.mpl_widget place
        self.setWindowTitle('Audio Analayzer')
        self.init_ui()

    def init_ui(self):  
        self.audio = get_input_devices()[0] # todo: refactor

        self.ring_buffer = RingBuffer(self.audio)
        self.audio_filter = AudioFilter(self.ring_buffer)

        self.ring_buffer = self.audio_filter.ring_buffer
        self.samplerate = self.ring_buffer.samplerate

        data = self.ring_buffer.get_data()
        arr, extent = self.create_spectrogram(data, self.samplerate)
        self.image = plt.imshow(
            arr, 
            animated=True, 
            extent=extent, 
            aspect='auto', 
            cmap=mpl.colormaps['inferno']
        )
        # plt.yscale("log")
        # plt.ylim([100, 14000])
        plt.ylim([0, 3000])
        self.fig.colorbar(self.image)

        self.thread = Thread(target=self.draw_spectrogram) 
        self.thread.start()

    def create_spectrogram(self, data, samplerate):
        Ndft = 512 * 8
        Ndft_over = Ndft
        noverl = round(Ndft*0.8)
        window = signal.windows.hamming(Ndft)

        Sxx, f, _ = plt.mlab.specgram(
            x=data[:,0],
            NFFT=Ndft_over,
            Fs=samplerate,
            window=window,
            noverlap=noverl,
            scale_by_freq=True,
            mode='psd'
        )
    
        xmin, xmax = 0, 10
        extent = xmin, xmax, f[0], f[-1]
        temp = 10 * np.log10(Sxx)
        temp[temp<-140] = -140
        temp[temp>-25] = 0
        arr = np.flipud(temp)

        return arr, extent
    
    def draw_spectrogram(self):
        with self.ring_buffer.get_stream():
            self.ring_buffer.update_ring_buffer()
            while True:
                self.update_spectrogram()
                time.sleep(0.001)

    def update_spectrogram(self):
        data = self.ring_buffer.get_data()
        if self.audio_filter.is_kiss_connected:
            self.audio_filter.perform_data(data)
        arr, _ = self.create_spectrogram(data, self.samplerate)
        self.image.set_array(arr)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

app = QtWidgets.QApplication([])
application = MyWindow()
application.show()
sys.exit(app.exec())
