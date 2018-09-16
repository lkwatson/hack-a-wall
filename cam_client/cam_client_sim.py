from socketIO_client_nexus import SocketIO, LoggingNamespace
import json
import time
import random
import math

sock_client = SocketIO('localhost', 3000, LoggingNamespace)

i = 0

while True:
    if i < 10:

        x_1, x_2 = (25*math.sin(i/10.) + 50), (10*math.sin(i/89.) + 20)
        y_1, y_2 = (10*math.cos(i/80.) + 10), (5*math.sin(i/70.) + 20)

        sock_client.emit('messages',
                         json.dumps([
                            {'person_id': 'Billy', 'x': x_1, 'y': y_1},
                            {'person_id': 'Tommy', 'x': x_2, 'y': y_2}
                         ]))
    else:
        x_1 = (25*math.sin(i/100) + 75)
        y_1 = (10*math.cos(i/80) + 8)
        sock_client.emit('messages',
                         json.dumps([
                            {'person_id': 'Billy', 'x': x_1, 'y': y_1},
                         ]))

    i += 1
    if i > 40:
        i = 0

    time.sleep(0.2 + 0.2*random.random())
