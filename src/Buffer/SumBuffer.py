import pyopencl as cl

class SumBuffer:
    def __init__(self, context):
        self.context = context

    def init(self, nbWorkGroups):
        mf = cl.mem_flags
        print "sumbuf"
        print self.context
        self.sum = cl.Buffer(self.context, mf.READ_WRITE, size=4 * nbWorkGroups)
