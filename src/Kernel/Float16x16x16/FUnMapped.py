from TInput import TInput
from FCommon import FCommon

class FUnMapped(FCommon, TInput):
    
    def init(self, T, U):
        TInput.init(self, T)
        FCommon.init(self, U)
