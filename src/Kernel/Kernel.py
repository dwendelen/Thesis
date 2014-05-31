import pyopencl as cl

class Kernel:

    contextQueue = None
    kernel = None
    time = None
    
    def __init__(self, contextQueue):
        self.contextQueue = contextQueue
    
    def getNbOperaties(self, I, R, n):
        raise NotImplementedError()
    
    def getNbWorkGroups(self, I, R, n):
        raise NotImplementedError()
        
    def getDataTransferZonderCache(self, I, R, n):
        raise NotImplementedError()
        
    def getDataTransferMetCache(self, I, R, n):
        raise NotImplementedError()
    
    def getBasicElements(self, I, R, n):
        raise NotImplementedError()
    
    def getName(self):
        raise NotImplementedError()
    
    def compile(self):
        file = open('../opencl/' + self.getName() + '.cl', 'r')
        self.program = cl.Program(self.contextQueue.context, file.read()).build()
        self.kernel = cl.Kernel(self.program, self.getName())
    
    def _createInitBuf(self, array):
        mf = cl.mem_flags
        return cl.Buffer(self.contextQueue.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=array)
    
    def _createReadWriteBuf(self, nbBytes):
        mf = cl.mem_flags
        return cl.Buffer(self.contextQueue.context, mf.READ_WRITE, size=nbBytes)
    
    def run(self):
        e = cl.enqueue_nd_range_kernel(self.contextQueue.queue, self.kernel, self.getGlobalSize(), self.getLocalSize())
        
        if(self.contextQueue.profile):
            e.wait()
            self.time = (e.profile.end - e.profile.start)/ 1000000.0
    
    def getGlobalSize(self):
        raise NotImplementedError()
    
    def getLocalSize(self):
        raise NotImplementedError()
