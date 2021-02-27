export TF_CFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
export TF_LFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )

g++ -std=c++11 -shared -D_GLIBCXX_USE_CXX11_ABI=0 -o my_matmul.so my_matmul.cc ${TF_CFLAGS[@]} -fPIC ${TF_LFLAGS[@]} 

