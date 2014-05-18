import pyopencl as cl
import numpy as np
import numpy.linalg as la

a = np.array([[201,202],[203,204],[205, 206]], dtype = np.float32)

print a[0][0]
print a[0][1]
print a[1][0]
print a[1][1]

print a.shape

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

mf = cl.mem_flags
a_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
b_buf = cl.Buffer(ctx, mf.WRITE_ONLY, a.nbytes)

prg = cl.Program(ctx, """
    __kernel void sum(__global float *a, __global float *b)
    {   
        int i = get_global_id(0);
        b[i] = a[i];
    }""").build()

prg.sum(queue, (6, 1), None, a_buf, b_buf)

c = np.array([0,0, 0,0, 0, 0], dtype = np.float32)
cl.enqueue_copy(queue, c, b_buf)

print c
