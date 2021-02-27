#define EIGEN_USE_THREADS

#include <utility>
#include <stack>
#include <algorithm>
#include <stdlib.h>
#include <math.h>
#include <string>
#include <vector>
#include "tensorflow/core/framework/tensor_reference.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/kernels/bounds_check.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"

void vector_add_kernel_luncher( const float* inputs, const float* scala, float* output, const int SIZE );

using namespace tensorflow;

REGISTER_OP("TensorVectorAdd")
.Input("in_tensor1: float")
.Input("in_scala: float")
.Output("out: float")
.SetShapeFn([](shape_inference::InferenceContext* c) {
        c->set_output(0, c->input(0)); return Status::OK();})
.Doc(R"doc( My Vector Add )doc");

class TensorVectorAddOp : public OpKernel
{
    public:
    ~TensorVectorAddOp() {}

    explicit TensorVectorAddOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}
    
    void Compute(OpKernelContext* ctx) override 
    {
        const Tensor& input_tensor = ctx->input(0);
        //const float* input_datas = reinterpret_cast<const float*>( input_tensor.flat<float>().data() );
        auto input_datas = input_tensor.flat<float>().data();
        const Tensor& scala_tensor = ctx->input(1);
        auto SCALA_VALUE = scala_tensor.flat<float>().data();

        Tensor* output_tensor = nullptr;
        OP_REQUIRES_OK(ctx, ctx->allocate_output(0, input_tensor.shape(), &output_tensor));
		//float* output = reinterpret_cast<float*>( output_tensor->flat<float>().data() );
		auto output = output_tensor->flat<float>().data();

        const int X = input_tensor.dim_size(0);
        const int Y = input_tensor.dim_size(1);
		const int SIZE = X * Y;
		printf("aaaa a\n");
		output[0] = 0.1;
		printf("bbbb\n");

		vector_add_kernel_luncher( input_datas, SCALA_VALUE, output, SIZE );
    }
};

REGISTER_KERNEL_BUILDER(Name("TensorVectorAdd").Device(DEVICE_GPU), TensorVectorAddOp);

