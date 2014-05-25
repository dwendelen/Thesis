import pyopencl as cl

class TempBuffer:
    def __init__(self, context):
        self.context = context

    def init(self, nbBytes):
        mf = cl.mem_flags
        self.sum = cl.Buffer(self.context, mf.READ_WRITE, size=nbBytes)
