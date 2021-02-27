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
        auto input_datas = input_tensor.flat<float>();

        const Tensor& scala_tensor = ctx->input(1);
        const float SCALA_VALUE = scala_tensor.flat<float>()(0);

        Tensor* output_tensor = nullptr;
        OP_REQUIRES_OK(ctx, ctx->allocate_output(0, input_tensor.shape(), &output_tensor));
		auto output = output_tensor->flat<float>();

        const int X = input_tensor.dim_size(0);
        const int Y = input_tensor.dim_size(1);
		const int size = X * Y;
		
		for( int i = 0; i < size; i++ )
			output(i) = input_datas(i) + SCALA_VALUE;
    }
};

REGISTER_KERNEL_BUILDER(Name("TensorVectorAdd").Device(DEVICE_CPU), TensorVectorAddOp);

