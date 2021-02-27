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

using namespace tensorflow;

//tensor_log()
REGISTER_OP("TensorLog")
.Input("in_tensor: float")
.Output("out: float")
.SetShapeFn([](shape_inference::InferenceContext* c) {
        c->set_output(0, c->input(0)); return Status::OK();})
.Doc(R"doc( log Tensor Value )doc");

class TensorLogOp : public OpKernel
{
    public:
    ~TensorLogOp() {}

    explicit TensorLogOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}
    
    void Compute(OpKernelContext* ctx) override 
    {
        const Tensor& input_tensor = ctx->input(0);
        auto input_datas = input_tensor.flat<float>();
		ctx->set_output(0, ctx->input(0));

		//int a = input_tensor.dims();
        const int X = input_tensor.dim_size(0);
        const int Y = input_tensor.dim_size(1);
		
        std::cout << "[TENSOR LOG START]" << std::endl;
        std::cout << "[TENSOR LOG] X : "<< X << std::endl;
        std::cout << "[TENSOR LOG] Y : "<< Y << std::endl;

        std::cout << "[TENSOR DATA]" << std::endl;
		const int size = X * Y;
		for( int i = 0; i < size; i++ )
        	std::cout << input_datas(i) << " ";

       	std::cout << std::endl;
        std::cout << "[TENSOR LOG END]" << std::endl;
    }
};

REGISTER_KERNEL_BUILDER(Name("TensorLog").Device(DEVICE_CPU), TensorLogOp);

