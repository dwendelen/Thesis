__attribute__((reqd_work_group_size(64, 1, 1)))
__kernel void Kernel(__global const float *T,
    __global const float *U1, __global const float *U2, __global const float *U3,
    const int R, __global float *sum)
{
	__local float l[64];
	
	float temp;
	float c = 0;
	
	int idxU1 = get_global_id(0);
	int idxU2 = get_global_id(1);
	int idxU3 = get_global_id(2);
	
	int I1 = get_global_size(0);
	int I2 = get_global_size(1);
	int I3 = get_global_size(2);
	
	for(int r = 0; r < R; r++)
	{
		temp = U1[idxU1];
        temp = temp * U2[idxU2];
        temp = temp * U3[idxU3];
     
        c += temp;
        
        idxU1 += I1;
        idxU2 += I2;
        idxU3 += I3;
	}
	
	float sum1;
	
	int idxT = get_global_id(0) + I1*get_global_id(1) + I1*I2*get_global_id(2);
    temp = c - T[idxT];
	sum1 = temp * temp;
	
	int lIdx = get_local_id(0) + 4 * get_local_id(1) + 16 * get_local_id(2);
	
	l[lIdx] = sum1;
	
	barrier(CLK_LOCAL_MEM_FENCE);
	if(lIdx == 0)
	{        
        #pragma unroll
        for(int i = 1; i < 64; i++)
        {
            sum1 += l[i];
        }
        
        gId = get_group_id(0) + get_num_groups(0) * (get_group_id(1) + get_num_groups(1) * get_group_id(2));
        sum[gId] = sum1;
	}
}
