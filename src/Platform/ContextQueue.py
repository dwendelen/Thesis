import pyopencl as cl

class ContextQueue:
    def init(self):
        devices = cl.get_platforms()[0].get_devices(cl.device_type.GPU)
        self.context = cl.Context([devices[0]])
        self.queue = cl.CommandQueue(self.context, properties=cl.command_queue_properties.PROFILING_ENABLE)