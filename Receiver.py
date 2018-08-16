#  -*- coding:utf-8 -*-

__author__ = 'yyp'


import zmq
import cv2
import threading


READY = 0
PROCESSING = 1


class Receiver:
    def __init__(self):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind('tcp://*:5555')

        # self.socket2 = self.context.socket(zmq.REP)
        # self.socket2.bind('tcp://*:5556')

        self.state = READY
        self.process = 0

    def handle_msg(self, msg):
        print 'receive msg:', msg
        if msg == 'b':
            if self.state == READY:
                self.socket.send_string('OK')
            else:
                self.socket.send_string('NA')
        if msg == 'c':
            if self.state == PROCESSING:
                self.process += 1
                self.socket.send_string(str(self.process))
            else:
                self.socket.send_string('NA')

    def handle_obj(self, obj):
        print 'receive python object'
        cv2.imwrite('rev_default.jpg', obj)
        print 'image_save'
        self.socket.send_string('OK')

    def main_loop(self):
        while True:
            if self.state == READY:
                print 'Ready...'
                print 'Waiting for connection...'
                print 'Connection confirmed'
                self.handle_msg(self.socket.recv_string())
                self.state = PROCESSING
            elif self.state == PROCESSING:
                self.handle_obj(self.socket.recv_pyobj())
                while self.process != 100:
                    self.handle_msg(self.socket.recv_string())
                self.state = READY


if __name__ == '__main__':
    r = Receiver()
    r.main_loop()
    # t1 = threading.Thread(target=r.msg_loop)
    # t2 = threading.Thread(target=r.obj_loop)
    # t1.start()
    # t2.start()
    # t1.join()
    # t2.join()
