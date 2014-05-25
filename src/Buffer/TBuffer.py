
from BlockPadder import blockPad
from Buffer import InputBuffer

class TBuffer(InputBuffer):
    def setT(self, T):
        if(len(T.shape) != 3):
            raise Exception("Illegal shape.")
        
        T1 = blockPad(T, [16,16,16])
        
        self._setBuf(T1)
    
    def getNbBytes(self):
        return self.T.size