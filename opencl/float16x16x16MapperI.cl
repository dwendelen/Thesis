__attribute__((reqd_work_group_size(4, 4, 4)))
__kernel void Kernel(__global const float4 *T, __global float4 *TMapped)
{	
    int I1 = get_global_size(0);
    int I2 = get_global_size(1);
    int I3 = get_global_size(2);
    
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

	int lIdx = get_local_id(0) + 4 * get_local_id(1) + 16 * get_local_id(2);
    int gIdx = get_group_id(0) + get_num_groups(0) * (get_group_id(1) + get_num_groups(1) * get_group_id(2));
    
    int channel = gIdx % 8;

    int idxTMapped =  8192 * (gIdx/8) + 16*channel + 128*lIdx;

    #pragma unroll
    for(int i1 = 0, j = 0; i1 < 4; i1++)
    {
        #pragma unroll
        for(int i2 = 0; i2 < 4; i2++, j++)
        {
            TMapped[idxTMapped] = T[idxT];
            
            idxTMapped++;
            idxT += jumpNextIdxTMode2;
        }
        idxT += jumpNextIdxTMode3;
    }
}
