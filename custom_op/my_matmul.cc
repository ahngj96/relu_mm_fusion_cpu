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
//#include "tensorflow/core/kernels/bounds_check.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace std;
using namespace tensorflow;

REGISTER_OP("TensorMatMul")
.Input("in_a: float")
.Input("in_b: float")
.Output("out: float")
.SetShapeFn(
		[](shape_inference::InferenceContext* c)
		{
            shape_inference::ShapeHandle a;
            TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &a));
            shape_inference::ShapeHandle b;
            TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &b));

            //DimensionHandle output_rows = c->Dim(a, 0);
            //DimensionHandle output_cols = c->Dim(b, 1);

            c->set_output(0, c->Matrix(c->Dim(a, 0), c->Dim(b, 1)));
            return Status::OK();
		}
)
.Doc(R"doc( My MatMul)doc");

class TensorMatMulOp : public OpKernel
{
    public:
    ~TensorMatMulOp() {}

    explicit TensorMatMulOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}
    
    void Compute(OpKernelContext* ctx) override 
    {
        const Tensor& input_tensor = ctx->input(0);
        auto a = input_tensor.tensor<float,2>();

        const Tensor& input_tensor2 = ctx->input(1);
        auto b = input_tensor2.tensor<float,2>();
        int64 A_shape[] = {input_tensor.dim_size(0), input_tensor.dim_size(1)};
        int64 B_shape[] = {input_tensor2.dim_size(0), input_tensor2.dim_size(1)};

        TensorShape C_shape = TensorShape({A_shape[0], B_shape[1]});
        //std::cout << "matmul input A shape " << A_shape[0] << ", "<< A_shape[1] <<std::endl;
        //std::cout << "matmul input b shape " << B_shape[0] << ", "<< B_shape[1] <<std::endl;
        //std::cout << "matmul output shape " << C_shape <<std::endl;

        Tensor* output_tensor = nullptr;
        OP_REQUIRES_OK(ctx, ctx->allocate_output(0, C_shape, &output_tensor));
		auto out = output_tensor->tensor<float,2>();

		//2x2
		int M = input_tensor.dim_size(0);
		int K = input_tensor.dim_size(1);
		int N = input_tensor2.dim_size(1);
        std::cout << "M K N: " << M << "," << K << ", " << N << std::endl;
        int count = 0;

		for( int m = 0; m < M; m++ )
		{
			for( int n = 0; n < N; n++ )
			{
				float sum = 0.0;
				for( int k = 0; k < K; k++ )
				{
					sum = sum + a(m,k) * b(k,n);
				}
				out(m,n) = sum;
                if(sum <= 0) count++;
				//TODO relu
			}
		}
        std::cout << "finished! -- under zero count: "<< count  << " / " << M *N << std::endl;
    }
};

REGISTER_KERNEL_BUILDER(Name("TensorMatMul").Device(DEVICE_CPU), TensorMatMulOp);

