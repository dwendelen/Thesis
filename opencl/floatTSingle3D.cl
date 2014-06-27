__attribute__((reqd_work_group_size(4, 4, 4)))
__kernel void float16x16x16(__global const float *T,
    __global const float *U0, __global const float *U1, __global const float *U2,
    int R, __global float *sum, const int I0, const int I1, const int I2)
{
	__local float l[64];
	
	float reg;
	float sum = 0;
	
	int id = get_global_id(0);
	int id0 = id % I0;
	int id1 = (id / I0) % I1;
	int id2 = id / (I0 * I1);
	
	for(int r = 0; r < R; r++)
	{
		reg = U0[id0];
        reg = reg * U1[id1];
        reg = reg * U2[id2];
     
        sum += reg;
        
        gId0 += I0;
        gId1 += I1;
        gId2 += I2;
	}
	
    reg = sum - T[idx];
	sum = reg * reg;
	
	int lId = get_local_id(0);
	
	l[lId] = sum;
	
	if(lId == 0)
	{        
        #pragma unroll
        for(int i = 1; i < 64; i++)
        {
            sum += l[i];
        }
        
        sum[gIdx] = sum;
	}
}