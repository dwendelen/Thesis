import pyopencl as cl

class Kernel:
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
    
    def setTBuffer(self, TBuffer):
        TBuffer.addKernel(self)
        self.updatedTBuffer(TBuffer)
    
    def setUBuffer(self, UBuffer):
        UBuffer.addKernel(self)
        self.updatedUBuffer(UBuffer)
    
    def updatedTBuffer(self, TBuffer):
        raise NotImplementedError()
    
    def updatedUBuffer(self, UBuffer):
        raise NotImplementedError()
    
    def run(self):
        e = cl.enqueue_nd_range_kernel(self.contextQueue.queue, self.kernel, self.getGlobalSize(), self.getLocalSize())
        self.time = (e.profile.end - e.profile.start)/ 1000000.0
    
    def getGlobalSize(self):
        raise NotImplementedError()
    
    def getLocalSize(self):
        raise NotImplementedError()
    
    def init(self):
        raise NotImplementedError()