# TARGET  := libGridsampleIPluginV2DynamicExt.so
# CC      := g++
# LIBS    := -lnvinfer
# LDFLAGS := -L/usr/lib/x86_64-linux-gnu/
# INCLUDE := -I. -I/usr/include/x86_64-linux-gnu/ -I/usr/local/cuda/include/
# CXXFLAGS:= -std=c++14 -Wall -O2 $(INCLUDE) -Wno-deprecated-declarations
# SHARE   := -fPIC -shared -o
# SOURCE  := $(wildcard *.c) $(wildcard *.cpp)

# .PHONY : clean

# all : $(TARGET)

# clean:
# 	rm -fr $(TARGET)

# $(TARGET) : $(SOURCE)
# 	$(CC) $(CXXFLAGS) $(SHARE) $@ $(SOURCE) $(LDFLAGS) $(LIBS)



CC=g++
LD=ld
CXXFLAGS=-Wall -std=c++14 -g -O

NVCC=nvcc

compute=86
NVCCFLAGS=-m64 -gencode arch=compute_$(compute),code=sm_$(compute) \
               -gencode arch=compute_$(compute),code=compute_$(compute)

# These are the directories where I installed TensorRT on my x86_64 PC.
TENSORRT_INCS=-I"/usr/include/x86_64-linux-gnu"
TENSORRT_LIBS=-L"/usr/lib/x86_64-linux-gnu"

# INCS and LIBS
INCS=-I"/usr/local/cuda/include" $(TENSORRT_INCS) -I"/usr/local/include" -I"plugin"
LIBS=-L"/usr/local/cuda/lib64" $(TENSORRT_LIBS) -L"/usr/local/lib" -Wl,--start-group -lnvinfer -lnvparsers -lnvinfer_plugin -lcudnn -lcublas -lcudart_static -lnvToolsExt -lcudart -lrt -ldl -lpthread -Wl,--end-group

.PHONY: all clean

CUSOURCE=gridSampler.cu

# This is the corresponding object file
CUOBJECT=gridSampler.o

all: gridSamplerPlugin.so

clean:
	rm -f *.so *.o

$(CUOBJECT): $(CUSOURCE)
	$(NVCC) -ccbin $(CC) $(INCS) $(NVCCFLAGS) -Xcompiler -fPIC -c -o $@ $<

gridSamplerPlugin.so: gridSamplerPlugin.o $(CUOBJECT)
	$(CC) -shared -o $@ $^ $(LIBS)

gridSamplerPlugin.o: GridsampleIPluginV2DynamicExt.cpp GridsampleIPluginV2DynamicExt.h
	$(NVCC) -ccbin $(CC) $(INCS) $(NVCCFLAGS) -Xcompiler -fPIC -c -o $@ $<
