import time

class timer():
    def __init__(self, log):
        self.log = log
    def __enter__(self):
        self.s = time.time()
    def __exit__(self, *argv):
        print(self.log, time.time() - self.s, ' sec')
