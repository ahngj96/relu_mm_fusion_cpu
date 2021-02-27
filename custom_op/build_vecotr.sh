export TF_CFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
export TF_LFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )

g++ -std=c++11 -shared -D_GLIBCXX_USE_CXX11_ABI=0 -o vector_add.so vector_add.cc ${TF_CFLAGS[@]} -fPIC ${TF_LFLAGS[@]} 

