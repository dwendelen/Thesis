import pyopencl as cl

class SumBuffer:
    def __init__(self, nbWorkGroups):
        mf = cl.mem_flags
        self.sum = cl.Buffer(self.context, mf.WRITE_ONLY, size=4 * nbWorkGroups)