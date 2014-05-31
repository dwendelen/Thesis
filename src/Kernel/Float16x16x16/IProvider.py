import numpy as np

class IProvider:
    I = (None, None, None)
    
    def init(self, I):
        self.I = I
