export TF_CFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
export TF_LFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )

nvcc -std=c++11 -c -D_GLIBCXX_USE_CXX11_ABI=0 -o vector_add_gpu.cu.o vector_add_gpu.cu.cc ${TF_CFLAGS[@]} -x cu -Xcompiler -fPIC -D GOOGLE_CUDA=1 -I/usr/local 

g++ -std=c++11 -shared -D_GLIBCXX_USE_CXX11_ABI=0 -o vector_add_gpu.so vector_add_gpu.cc vector_add_gpu.cu.o ${TF_CFLAGS[@]} -fPIC -lcudart ${TF_LFLAGS[@]} -L/usr/local/cuda/lib64 -D GOOGLE_CUDA=1

