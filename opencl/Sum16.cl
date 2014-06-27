#pragma OPENCL EXTENSION cl_amd_printf : enable

#define NUM_WI   16

__attribute__((reqd_work_group_size(NUM_WI,1,1)))
__kernel void Sum16(__global const float4 *array, const int n1, __global float4 *sum)
{ 	
    int n = n1/4;
    __local float4 l[NUM_WI];
    int idx = get_global_id(0);
    int jump = get_num_groups(0) * NUM_WI;
    
    float4 s = 0.0f;
    
    for(idx = 0; idx < n; idx += jump)
    {
        s += array[idx];
    }
	
	
	l[get_local_id(0)] = s;
	
	if(get_local_id(0) == 0)
	{   
	    #pragma unroll
	    for(int i = 1; i < NUM_WI; i++)
	    {
	        s += l[i];
	    }
	    sum[get_group_id(0)] = s;
	}
	
	
}
