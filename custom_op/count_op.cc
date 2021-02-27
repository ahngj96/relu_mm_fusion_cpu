#define EIGEN_USE_THREADS

#include <utility>
#include <stack>
#include <algorithm>
#include <stdlib.h>
#include <math.h>
#include <string>
#include <vector>
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_reference.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/kernels/bounds_check.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace std;
using namespace tensorflow;
using CPUDevice = Eigen::ThreadPoolDevice;

REGISTER_OP("TensorCount")
.Input("in_tensor: float")
.Output("out: int32")
.Doc(R"doc( my count op )doc");

class TensorCountOp : public OpKernel
{
	private:
	const float TH = 1.1;

    public:
    ~TensorCountOp() {}

    explicit TensorCountOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}
    
    void Compute(OpKernelContext* ctx) override 
    {
        const Tensor& input_tensor = ctx->input(0);
        auto input_datas = input_tensor.flat<float>();

        Tensor* output_tensor = nullptr;
		TensorShape shape_({1,1});
        OP_REQUIRES_OK(ctx, ctx->allocate_output(0, shape_, &output_tensor));
		auto output = output_tensor->flat<int>();

        const int X = input_tensor.dim_size(0);
        const int Y = input_tensor.dim_size(1);
		const int size = X * Y;
		
		unsigned int cnt = 0;
		for( int i = 0; i < size; i++ )
		{
			if( input_datas(i) < TH )
				cnt++;
		}

		output(0) = cnt;
    }
};

REGISTER_KERNEL_BUILDER(Name("TensorCount").Device(DEVICE_CPU), TensorCountOp);

