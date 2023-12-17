GCC  = g++
NVCC = nvcc

CUDA_DIR = $(CUDA_HOME)
CUDA_LIB = -L$(CUDA_DIR)/lib64
CUDA_INC = -I$(CUDA_DIR)/include
CUDA_INC_COMMON = -I../Common

C_FLAGS = -O3
NVCC_FLAGS = -m64 --std=c++11
LD_LIBS	= -lcudart -lm
GENCODE_FLAGS = -Wno-deprecated-gpu-targets

FILE = bandwidthTest

$(FILE): $(FILE).o
	 $(GCC) $(C_FLAGS) $< -o $(FILE) $(CUDA_LIB) $(LD_LIBS)

$(FILE).o: $(FILE).cu
	   $(NVCC) $(NVCC_FLAGS) $(CUDA_INC) $(CUDA_INC_COMMON) $(GENCODE_FLAGS) -o $@ -c $<

run: 
	./$(FILE) --device=0 --memory=pinned --mode=range --start=1024 --end=67108864 --increment=524288 --htod --csv
	./$(FILE) --device=0 --memory=pinned --mode=range --start=1024 --end=67108864 --increment=524288 --dtod --csv
	./$(FILE) --device=0 --memory=pinned --mode=range --start=1024 --end=67108864 --increment=524288 --dtoh --csv

clean:
	rm -f *.o $(FILE)

