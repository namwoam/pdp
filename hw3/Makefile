NVCC        := nvcc
SRC         := main.cu
TARGET      := main
LDFLAGS     := -lcublas
NVCC_FLAGS  := -O3 -std=c++14
CUDA_ARCH   := 70

CUDA_SRCS := $(shell find . -type f -name '*.cu')
HEADERS   := $(shell find . -type f -name '*.h')

.PHONY: all default build run bench figures clean

all: build
default: build

build: $(TARGET)

$(TARGET): $(CUDA_SRCS) $(HEADERS)
	$(NVCC) $(NVCC_FLAGS) \
		-gencode=arch=compute_$(CUDA_ARCH),code=sm_$(CUDA_ARCH) \
		$(SRC) -o $(TARGET) $(LDFLAGS)

run: build
	./$(TARGET) 1

bench: build
	sh ./bench.sh

figures:
	poetry run python scripts/generate_report_assets.py

clean:
	rm -f $(TARGET)