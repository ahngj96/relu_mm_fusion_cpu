#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

//#include <cuda.h>

__global__ void add_kernel( const float* inputs, const float* scala, float* output, const int SIZE )
{
	int idx = threadIdx.x;
	if( idx >= SIZE )
		return;

	const float SC = scala[0];
	output[idx] = inputs[idx] + SC;
}

void vector_add_kernel_luncher( const float* inputs, const float* scala, float* output, const int SIZE ) 
{
	add_kernel<<<1,256>>>( inputs, scala, output, SIZE );
}

#endif
