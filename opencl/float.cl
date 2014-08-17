__kernel void Kernel(__global const float* T,
    __global const float* U1, __global const float* U2, __global const float* U3,
    const int R, __global float* sum, const int I1, const int I2, const int I3, local float* l)
{
	float temp;
	float c;
	
	int idxT = get_global_id(0);
	int idxU1 = idxT % I1;
	int idxU2 = (idxT / I1) % I2;
	int idxU3 = idxT / (I1 * I2);
	
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
	
    temp = c - T[idxT];
	sum1 = temp * temp;
	
	l[get_local_id(0)] = sum1;
	
	barrier(CLK_LOCAL_MEM_FENCE);
	if(lIdx == 0)
	{        
        for(int i = 1; i < get_local_size(0); i++)
        {
            sum1 += l[i];
        }
        
        sum[get_group_id(0)] = sum1;
	}
}
