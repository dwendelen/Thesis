import pyopencl as cl
from BlockPadder import blockPad
import numpy as np

class Buffer:

    def __init__(self, context):
        self.context = context
        self.kernels = []
        
    def addKernel(self, kernel):
        self.kernels.append(kernel)