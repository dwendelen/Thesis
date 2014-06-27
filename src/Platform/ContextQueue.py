import pyopencl as cl

class ContextQueue:

    profile = None
    queue = None
    context = None
    
    def __init__(self, profile = False):
        self.profile = profile

    def init(self):
        devices = cl.get_platforms()[0].get_devices(cl.device_type.CPU)
        self.context = cl.Context([devices[0]])
        if(self.profile):
            self.queue = cl.CommandQueue(self.context, properties=cl.command_queue_properties.PROFILING_ENABLE)
        else:
            self.queue = cl.CommandQueue(self.context)
