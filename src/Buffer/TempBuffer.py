import pyopencl as cl
from Buffer import Buffer

class TempBuffer(Buffer):
    def __init__(self, context):
        self.context = context

    def init(self, nbBytes):
        mf = cl.mem_flags
        self._buffer = cl.Buffer(self.context, mf.READ_WRITE, size=nbBytes)
    
class TempBufferFactory():
    def createWithSameSize(self, buf):
        t = TempBuffer()
        t.init(buf.getNbBytes())
        return t
