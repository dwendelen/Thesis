import pyopencl as cl

class Buffer:
    _context = None
    _buffer = None
    _kernels = []
    
    def __init__(self, context):
        self._context = context

    def addKernel(self, kernel):
        self._kernels.append(kernel)
    
    def getBuffer(self):
        return self._buffer
        
    def getNbBytes(self):
        return self.getBuffer().size

class InputBuffer(Buffer):
    def _setBuf(self, array):
        mf = cl.mem_flags
        
        self._buffer cl.Buffer(self._context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=array)
        
        for kernel in self._kernels:
            kernel.setTBuffer(self)