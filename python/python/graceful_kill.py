# coding: utf8
import signal
import time

class GracefulKiller(object):
    kill_now = False

    def __init__(self):
        # signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)

    def exit_gracefully(self, signum, frame):
        self.kill_now = True

if __name__ == '__main__':
    killer = GracefulKiller()
    while True:
        time.sleep(1)
        print('do something')
        if killer.kill_now:
            print('gracefully exit')
            exit(0)