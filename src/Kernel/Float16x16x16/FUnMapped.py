from TInput import TInput
from FCommon import FCommon

class FUnMapped(FCommon, TInput):
        
    def init(self, T, R, U, I, Sum):
        FCommon.init(self, R, U, I, Sum)
        TInput.init(self, T)
