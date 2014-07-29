#pragma OPENCL EXTENSION cl_amd_printf : enable

__attribute__((reqd_work_group_size(4, 4, 4)))
__kernel void Kernel(__global const double2 *T,
    __global const double2 *U0, __global const double2 *U1, __global const double2 *U2,
    int R, __global double *sum)
{   
    __local double l[128];
    
    double2 a;
    double2 b[4];
    double2 c[16];
    double2 f;
    
    int gIdx = get_group_id(0) + get_num_groups(0) * (get_group_id(1) + get_num_groups(1) * get_group_id(2));
    int channel = gIdx % 8;
    
    int I0 = get_global_size(0);
    int I1 = get_global_size(1);
    int I2 = get_global_size(2);
    
    int jump0 = 8 * I0;
    int jump1 = 8 * I1;
    int jump2 = 8 * I2;
    
    #pragma unroll
    for(int i = 0; i < 16; i++)
    {
        c[i] = 0;
    }
    
    int gId0 = get_global_id(0);
    int gId1 = get_global_id(1);
    int gId2 = get_global_id(2);

	gId0 = (gId0/18) * 128 + channel * 16 + gId0 % 16;
	gId1 = (gId1/18) * 128 + channel * 16 + gId1 % 16;
	gId2 = (gId2/18) * 128 + channel * 16 + gId2 % 16;

    for(int r = 0; r < R; r++)
    {   
        //Fetch eerste 16 met n=0
        a = U0[gId0];

        //barrier(CLK_LOCAL_MEM_FENCE);
        
        //Fetch volgende 16 met n=1
        f = U1[gId1];
        
        b[0] = a * f.x;
        b[1] = a * f.y;
        b[2] = a * f.z;
        b[3] = a * f.w;

        //barrier(CLK_LOCAL_MEM_FENCE);
        
        //Fetch laatste 16 met n=2
        f = U2[gId2];
        
        #pragma unroll
        for(int i = 0; i < 4; i++)
        {
            c[i +  0] += b[i] * f.x;
            c[i +  4] += b[i] * f.y;
            c[i +  8] += b[i] * f.z;
            c[i + 12] += b[i] * f.w;
        }
        
        gId0 += jump0;
        gId1 += jump1;
        gId2 += jump2;
    }
    //T-waarde aftrekken
    int gIdx0 = get_global_id(0);
    int gIdx1 = get_global_id(1);
    int gIdx2 = get_global_id(2);
    
    int lIdx = get_local_id(0) + 4 * get_local_id(1) + 16 * get_local_id(2);
    
    //Calculate first index
    int idx =  8*1024 * (gIdx/8) + 16*channel + 128*lIdx;   

	double2 s = 0;
	double2 t;

    #pragma unroll
    for(int i = 0; i < 16; i++)
    {
        //Handle the 4 floats along the 0-axis
        f = T[idx];
        t = c[i] - f;
		s += t*t;
			
        idx++;
    }

    l[lIdx] = s.x + s.y + s.z + s.w;
    
    barrier(CLK_LOCAL_MEM_FENCE);
    
    if(lIdx == 0)
    {
        double sss = 0;
        #pragma unroll
        for(int i = 0; i < 64; i++)
        {
            sss += l[i];
        }
        
        sum[gIdx] = sss;
    }
}