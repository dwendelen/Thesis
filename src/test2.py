import pyopencl as cl
import numpy as np
import numpy.linalg as la

a = np.array([[201,202],[203,204]], dtype = np.float32)

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

mf = cl.mem_flags
a_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
b_buf = cl.Buffer(ctx, mf.WRITE_ONLY, a.nbytes)

prg = cl.Program(ctx, """
    #pragma OPENCL EXTENSION cl_amd_printf: enable
    __kernel void k1(__global float *a, __global float *b)
    {   
        int i = get_global_id(0);
        //printf("%f :: %d\\n", a[i], i);
        b[i] = a[i]
    }""").build()

prg.sum(queue, a.shape, None, a_buf, b_buf)

c = np.zeros(4)
cl.enqueue_copy(queue, c, b_buf)
