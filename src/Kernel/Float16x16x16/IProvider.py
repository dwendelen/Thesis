class IProvider:
    I = (None, None, None)
    
    def getI(self):
        return self.I
    
    def initFromIProvider(self, iProvider):
        self.I = iProvider.I
    
    def init(self, I):
        self.I = I