__attribute__((reqd_work_group_size(4, 4, 4)))
__kernel void Kernel(__global const float2 *T,
    __global const float2 *U1, __global const float2 *U2, __global const float2 *U3,
    const int R, __global float *sum)
{   
    __local float l[64];
    
    float2 u1;
    float2 u1u2[2];
    float2 c[4];
    float2 temp;
    
    int I1 = get_global_size(0);
    int I2 = get_global_size(1);
    int I3 = get_global_size(2);
    
    #pragma unroll
    for(int i = 0; i < 4; i++)
    {
        c[i] = 0;
    }
    
    int idxU1 = get_global_id(0);
    int idxU2 = get_global_id(1);
    int idxU3 = get_global_id(2);

    for(int r = 0; r < R; r++)
    {   
        u1 = U1[idxU1];

        temp = U2[idxU2];
        
        u1u2[0] = u1 * temp.x;
        u1u2[1] = u1 * temp.y;

        temp = U3[idxU3];
        
        c[0] += u1u2[0] * temp.x;
        c[1] += u1u2[1] * temp.x;
        c[2] += u1u2[0] * temp.y;
        c[3] += u1u2[1] * temp.y;
        
        idxU1 += I1;
        idxU2 += I2;
        idxU3 += I3;
    }
    int gId1 = get_global_id(0);
    int gId2 = get_global_id(1);
    int gId3 = get_global_id(2);
    
    int jumpIdxTMode2 = I1;
    int jumpIdxTMode3 = 2*I1*I2;
    
    int idxT = gId1 + 
        2*gId2 * jumpIdxTMode2 +
        2*gId3 * jumpIdxTMode3;

    int jumpNextIdxTMode2 = jumpIdxTMode2;
	int jumpNextIdxTMode3 = jumpIdxTMode3 - 2*jumpIdxTMode2;

	float2 sum2 = 0;

    #pragma unroll
    for(int i1 = 0, j = 0; i1 < 2; i1++)
    {
        #pragma unroll
        for(int i2 = 0; i2 < 2; i2++, j++)
        {
            temp = c[j] - T[idxT];
			sum2 += temp*temp;
			
            idxT += jumpNextIdxTMode2;
        }
        idxT += jumpNextIdxTMode3;
    }

    float sum1 = sum2.x + sum2.y;
    
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
