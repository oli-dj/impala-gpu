
__global__ void multiplyArray(const int* a, const int* b, const int N, const int M, int* c){
    /*
    Embarrassingly simple parallel multiply where output and first matrix are 1D
     
    Inputs:
    a: N length array of matches (ideally ones and zeros)
    b: N*M length array of counts (integers)
    N: listLenght
    M: numFacies


    Input/Output:
    c: Result of element-wise multiplication of 'a' with each "row" of 'b'.
    */
    
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if(i < N){
        for(int j = 0; j < M; j++){
            c[M*i + j] = a[i] * b[M*i + j];
        }
    }
} 