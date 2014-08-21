__attribute__((reqd_work_group_size(4, 4, 4)))
__kernel void Kernel(__global const float4* T,
    __global const float4* U1, __global const float4* U2, __global const float4* U3,
    const int R, __global float* sum)
{   
    __local float l[64];
    
    float4 u1;
    float4 u1u2[4];
    float4 c[16];
    float4 temp;
    
    int I1 = get_global_size(0);
    int I2 = get_global_size(1);
    int I3 = get_global_size(2);
    
    #pragma unroll
    for(int i = 0; i < 16; i++)
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
        u1u2[2] = u1 * temp.z;
        u1u2[3] = u1 * temp.w;

        temp = U3[idxU3];
        
        #pragma unroll
        for(int i = 0; i < 4; i++)
        {
            c[i +  0] += u1u2[i] * temp.x;
            c[i +  4] += u1u2[i] * temp.y;
            c[i +  8] += u1u2[i] * temp.z;
            c[i + 12] += u1u2[i] * temp.w;
        }
        
        idxU1 += I1;
        idxU2 += I2;
        idxU3 += I3;
    }
    int gId1 = get_global_id(0);
    int gId2 = get_global_id(1);
    int gId3 = get_global_id(2);
    
    int jumpIdxTMode2 = I1;
    int jumpIdxTMode3 = 4*I1*I2;
    
    int idxT = gId1 + 
        4*gId2 * jumpIdxTMode2 +
        4*gId3 * jumpIdxTMode3;

    int jumpNextIdxTMode2 = jumpIdxTMode2;
	int jumpNextIdxTMode3 = jumpIdxTMode3 - 4*jumpIdxTMode2;

	float4 sum4 = 0;

    #pragma unroll
    for(int i1 = 0, j = 0; i1 < 4; i1++)
    {
        #pragma unroll
        for(int i2 = 0; i2 < 4; i2++, j++)
        {
            temp = c[j] - T[idxT];
			sum4 += temp*temp;
			
            idxT += jumpNextIdxTMode2;
        }
        idxT += jumpNextIdxTMode3;
    }

    float sum1 = sum4.x + sum4.y + sum4.z + sum4.w;
    
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
        
        int gIdx = get_group_id(0) + get_num_groups(0) * (get_group_id(1) + get_num_groups(1) * get_group_id(2));
        sum[gIdx] = sum1;
	}
}
