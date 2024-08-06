import socket
import time
import json

HOST = "127.0.0.1"  # The server's hostname or IP address
PORT = 6025  # The port used by the server

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.connect((HOST, PORT))
    while True:
        sock.sendall(json.dumps({ "mode": "uplink" }).encode('utf-8'))
        time.sleep(0.5)
        sock.sendall(json.dumps({ "mode": "dnlink" }).encode('utf-8'))
        time.sleep(2.5)