#pragma OPENCL EXTENSION cl_amd_printf : enable

//The number of consecutive float4s served by one channel
#define OFFSET      64

//LEN = OFFSET * nbWorkItems (=16)
#define LEN		  1024

//JUMP = LEN - OFFSET
#define JUMP       960

/*
Basic idea:
To have two workitems per channel, to keep the channels busy.
*/
__attribute__((reqd_work_group_size(16,1,1)))
__kernel void Sum16(__global const float *array, const int n, __global float *sum)
{   
	__local float l[16];
	
	const int limit1 = (n / LEN) * LEN;
	const int limit2 = (n / 4) * 4;
	int idx = get_local_id(0) * OFFSET;
	float4 s;
	
	if(get_local_id(0) == 0)
	{
	    printf("limit1: %i, limit2: %i", limit1, limit2);
	}
	
	//Sum up a whole block
	while(idx < limit1)
	{
		#pragma unroll
		for(int i = 0; i < OFFSET; i++){
			s.x += array[idx + 0];
			s.y += array[idx + 1];
			s.z += array[idx + 2];
			s.w += array[idx + 3];
			idx += 4;
		}
		
		idx += JUMP;
	}
	
	//Sum up the remaining float4s
	while(idx < limit2)
	{
		s.x += array[idx + 0];
		s.y += array[idx + 1];
		s.z += array[idx + 2];
		s.w += array[idx + 3];
		idx += 4;
	}
	
	s.x += s.z;
	s.y += s.w;
	s.x += s.y;
	
	//Sum up the remaining floats
	while(idx < n)
	{
		s.x += array[idx];
		idx++;
	}
	
	l[get_local_id(0)] = s.x;
	
	if(get_local_id(0) == 0)
	{
		#pragma unroll
		for(int i = 1; i < 16; i++)
		{
			s.x += l[i];
		}
		
		sum[0] = s.x;
	}
}
