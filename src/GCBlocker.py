class GCBlocker:
    '''Add buffers to this class to prevent them from
    being garbagecollected.
    !!!MAKE SURE THIS CLASS WILL NOT BE GARBAGECOLLECTED'''
    
    def __init__(self):
        self.reg = list()
    
    def remember(self, buf):
        self.reg.append(buf)
        
    def forgetAll(self):
        self.reg.clear()