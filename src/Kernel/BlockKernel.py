from Kernel import Kernel

class Block3DKernel(Kernel):
    def getBlockSize(self):
        raise NotImplementedError()
    
    def getGlobalSize(self):
        return ( self.I[0]/self.getLocalSize()[0],
                 self.I[1]/self.getLocalSize()[1],
                 self.I[2]/self.getLocalSize()[2])