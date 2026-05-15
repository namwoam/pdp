NVCC       := nvcc
SRC        := main.cu
TARGET     := main
LDFLAGS    := -lcublas

NVCC_FLAGS := -O3 -std=c++14

# V100 = sm_70. Override with `make CUDA_ARCH=80` etc. if needed.
CUDA_ARCH ?= 70
NVCC_FLAGS += -gencode=arch=compute_$(CUDA_ARCH),code=sm_$(CUDA_ARCH)

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCC_FLAGS) $(SRC) -o $(TARGET) $(LDFLAGS)

clean:
	rm -f $(TARGET)

.PHONY: all clean
