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
        raise NotImplementedError()
    
    def setUBuffer(self, UBuffer):
        raise NotImplementedError()
    
    def run(self):
        raise NotImplementedError()
    
    def init(self):
        raise NotImplementedError()