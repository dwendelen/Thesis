import pyopencl as cl
import numpy as np
import numpy.linalg as la

a = np.array([[201,202],[203,204]], dtype = np.float32)

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

mf = cl.mem_flags
a_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)

prg = cl.Program(ctx, step1 = """
    #pragma OPENCL EXTENSION cl_amd_printf: enable
    __kernel void k1(__global float *a)
    {   
        int i = get_global_id(0);
        printf("%f :: %d\\n", a[i], i);
    }""").build()

prg.sum(queue, a.shape, None, a_buf)