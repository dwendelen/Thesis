__attribute__((reqd_work_group_size(8, 8, 1))) 
__kernel void KernelG1(__global const double2 *F, __global double2 *G1, __global const double2 *U2, __global const double2 *U3, const int I2, const int I3)
{
    double2 temp;
    double2 u2;
    
    double2 U3Cache[32];
    int sizeCache;
    
    double2 sum = 0;
    
    int idxTFirstCacheItem = get_global_id(1);
    int idxT;
    int idxR = get_global_id(0);
    int I1 = get_global_size(1);
    
    int startIdxU2 = idxR * I2;
    int idxU2;
    int idxU3 = idxR * I3;
    
    int limitU2 = startIdxU2 + I2;
    int limitU3 = idxU3 + I3;
    
    int jumpIdxTMode2 = I1;
    int jumpIdxTMode3 = 2 * I1 * I2;
    
    //  jumpIdxTForNextCache = 64*jumpIdxTMode3 - 2*jumpIdxTMode2*I2
    //  jumpIdxTForNextCache = 64*jumpIdxTMode3 - 2*I1*I2
    //  jumpIdxTForNextCache = 64*jumpIdxTMode3 - jumpIdxTMode3
    int jumpIdxTForNextCache = 63*jumpIdxTMode3;
    
    while(idxU3 < limitU3)
    {
    	sizeCache = 0;
    	int idxCache = 0;
    	
    	//Cache 64 doubles in blocks of 16, if available
    	#pragma unroll
    	for(int i = 0; i < 4; i++)
    	{	
    		if(idxU3 >= limitU3)
    			break;
    			
	    	#pragma unroll
	    	for(int j = 0; j < 8; j++)
	    	{
	        	U3Cache[idxCache++] = U3[idxU3++];
	    	}
        	sizeCache += 8;
        }
        
        for(idxU2 = startIdxU2; idxU2 < limitU2; idxU2++)
        {
            u2 = U2[idxU2];
            
            idxT = idxTFirstCacheItem;
            for(idxCache = 0; idxCache < sizeCache; )
            {
            	#pragma unroll
            	for(int j = 0; j < 8; j++)
		    	{
		    	    temp = u2 * U3Cache[idxCache++].y;
		        	sum += temp.x * F[idxT];
		        	sum += temp.y * F[idxT + jumpIdxTMode2];
		        	idxT += jumpIdxTMode3;
		        	
		        	temp = u2 * U3Cache[idxCache++].y;
		        	sum += temp.x * F[idxT];
					sum += temp.y * F[idxT + jumpIdxTMode2];
            		idxT += jumpIdxTMode3;
		    	}
            }
            idxTFirstCacheItem += 2*jumpIdxTMode2;
        }
        idxTFirstCacheItem += jumpIdxTForNextCache;
    }
    
    //  idxG1 =       idxT1      +         I0         *     idxR
    int idxG1 = get_global_id(1) + get_global_size(1) * get_global_id(0);
    G1[idxG1] = sum;
}

__attribute__((reqd_work_group_size(8, 8, 1)))
__kernel void KernelG2(__global const double2 *F, __global const double2 *U1, __global double2 *G2, __global const double2 *U3, const int I1, const int I3)
{
    double2 temp;
    double2 u1;
    
    double2 U3Cache[32];
    int sizeCache;
    
    double4 sum = 0;
    
    int idxR = get_global_id(0);
    int I2 = get_global_size(1);
    
    int jumpIdxTMode2 = I1;
    int jumpIdxTMode3 = 2 * I1 * I2;
    
    int idxTFirstCacheItem = get_global_id(1) * 2 * jumpIdxTMode2;
    int idxT;
    
    int startIdxU1 = idxR * I1;
    int idxU1;
    int idxU3 = idxR * I3;
    
    int limitU1 = startIdxU1 + I1;
    int limitU3 = idxU3 + I3;
    
    int jumpIdxTForNextCache = 64*jumpIdxTMode3 - I1;
    
    while(idxU3 < limitU3)
    {
    	sizeCache = 0;
    	int idxCache = 0;
    	
    	//Cache 64 doubles in blocks of 16, if available
    	#pragma unroll
    	for(int i = 0; i < 4; i++)
    	{	
    		if(idxU3 >= limitU3)
    			break;
    			
	    	#pragma unroll
	    	for(int j = 0; j < 8; j++)
	    	{
	        	U3Cache[idxCache++] = U3[idxU3++];
	    	}
        	sizeCache += 8;
        }
        
        for(idxU1 = startIdxU1; idxU1 < limitU1; idxU1++)
        {
            u1 = U1[idxU1];
            
            idxT = idxTFirstCacheItem;
            for(idxCache = 0; idxCache < sizeCache; )
            {
            	#pragma unroll
            	for(int j = 0; j < 8; j++)
		    	{
		        	temp = u1 * U3Cache[idxCache].x;
		        	sum.xz += temp * F[idxT];
		        	sum.yw += temp * F[idxT + jumpIdxTMode2];
		        	idxT += jumpIdxTMode3;
		        	
		        	temp = u1 * U3Cache[idxCache++].y;
		        	sum.xz += temp * F[idxT];
					sum.yw += temp * F[idxT + jumpIdxTMode2];
            		idxT += jumpIdxTMode3;
		    	}
            }
            idxTFirstCacheItem++;
        }
        idxTFirstCacheItem += jumpIdxTForNextCache;
    }
    
    //  idxG2 =       idxT2      +         I2         *     idxR
    int idxG2 = get_global_id(1) + get_global_size(1) * get_global_id(0);
    G2[idxG2] = sum.xy + sum.zw;
}
    
    __attribute__((reqd_work_group_size(8, 8, 1)))
__kernel void KernelG3(__global const double2 *F, __global const double2 *U1, __global const double2 *U2, __global double2 *G3, const int I1, const int I2)
{
    double2 temp;
    double2 u1;
    
    double2 U2Cache[32];
    int sizeCache;
    
    double2 sum = 0;
    
    int idxR = get_global_id(0);
    int I3 = get_global_size(1);
    
    int jumpIdxTMode2 = I1;
    int jumpIdxTMode3 = 2 * I1 * I2;
    
    int idxTFirstCacheItem = get_global_id(1)  * jumpIdxTMode3;
    int idxT;
    
    int startIdxU1 = idxR * I1;
    int idxU1;
    int idxU2 = idxR * I2;
    
    int limitU1 = startIdxU1 + I1;
    int limitU2 = idxU2 + I2;
    
    //  jumpIdxTForNextCache = 64*jumpIdxTMode2 - I1
    //  jumpIdxTForNextCache = 64*jumpIdxTMode2 - jumpIdxTMode2
    int jumpIdxTForNextCache = 63*jumpIdxTMode2;
    
    while(idxU2 < limitU2)
    {
    	sizeCache = 0;
    	int idxCache = 0;
    	
    	//Cache 64 doubles in blocks of 16, if available
    	#pragma unroll
    	for(int i = 0; i < 4; i++)
    	{	
    		if(idxU2 >= limitU2)
    			break;
    			
	    	#pragma unroll
	    	for(int j = 0; j < 8; j++)
	    	{
	        	U2Cache[idxCache++] = U2[idxU2++];
	    	}
        	sizeCache += 8;
        }
        
        for(idxU1 = startIdxU1; idxU1 < limitU1; idxU1++)
        {
            u1 = U1[idxU1];
            
            idxT = idxTFirstCacheItem;
            for(idxCache = 0; idxCache < sizeCache; )
            {
            	#pragma unroll
            	for(int j = 0; j < 8; j++)
		    	{
		        	temp = u1 * U2Cache[idxCache].x;
		        	sum += temp * F[idxT];
		        	sum += temp * F[idxT + jumpIdxTMode3];
		        	idxT += jumpIdxTMode2;
		        	
		        	temp = u1 * U2Cache[idxCache++].y;
		        	sum += temp * F[idxT];
					sum += temp * F[idxT + jumpIdxTMode3];
            		idxT += jumpIdxTMode2;
		    	}
            }
            idxTFirstCacheItem++;
        }
        idxTFirstCacheItem += jumpIdxTForNextCache;
    }
    
    //  idxG3 =       idxT3      +         I3         *     idxR
    int idxG3 = get_global_id(1) + get_global_size(1) * get_global_id(0);
    G3[idxG3] = sum;
}
