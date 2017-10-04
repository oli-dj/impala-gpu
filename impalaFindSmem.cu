
// #include <math.h> //necesarry?
// Define a NaN value
#define CUDART_NAN_F            __int_as_float(0x7fffffff)

__global__ void impalaFindSmem(float *list, float *dataEvent, int listLength, int dataLength, int *matches)
{
        int x = blockIdx.x*blockDim.x + threadIdx.x;
        int temp = 1;

        //TODO: Implement shared memory

        if (x < listLength)
        {
            for(int y=0;y<dataLength;y++){
                if(( !isnan(list[x+y])) && (list[x+y] != dataEvent[y]))
                {
                    temp = 0;
                }
            }
            matches[x] = temp;
        }
}