CC=g++
NVCC=nvcc
INCLUDE_DIR = -I/usr/local/cuda/include -I/usr/include/x86_64-linux-gnu
LIB_DIR = -L/usr/lib/x86_64-linux-gnu -L/usr/local/cuda/lib64  # <-- Add this path
LIBS = -lnvinfer -lnvparsers -lnvinfer_plugin -lcudart -ldl

all: trt_tester

trt_tester: trt_tester.cpp
	$(CC) -std=c++11 trt_tester.cpp -o trt_tester $(INCLUDE_DIR) $(LIB_DIR) $(LIBS)
