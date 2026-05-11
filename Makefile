# Makefile for GPU Portfolio Optimisation
# ---------------------------------------------------------------
#  Targets:
#    make all          →  builds cpu_baseline  and  portfolio_gpu
#    make cpu          →  builds cpu_baseline only
#    make gpu          →  builds portfolio_gpu only
#    make clean        →  removes object files and binaries
#    make benchmark    →  runs the Python benchmark script
#    make fetch_data   →  downloads returns CSV via fetch_data.py
#
#  Requirements:
#    CUDA toolkit  (nvcc)
#    cuBLAS        (usually bundled with CUDA)
#    Thrust        (bundled with CUDA ≥ 10)
#    C++17 compiler
# ---------------------------------------------------------------

# ---------- Compiler settings ----------
NVCC       := nvcc
CXX        := g++
NVCC_FLAGS := -O3 -std=c++17 -arch=sm_75 \
              --expt-relaxed-constexpr \
              -Xcompiler "-O3 -std=c++17 -Wall"
CXX_FLAGS  := -O3 -std=c++17 -Wall

# ---------- CUDA / cuBLAS ----------
# On Cray/NVHPC systems cuBLAS lives under math_libs, not cuda/lib64
NVHPC_ROOT := /opt/nvidia/hpc_sdk/Linux_x86_64/25.3
CUDA_HOME  ?= $(NVHPC_ROOT)/cuda/12.8
CUBLAS_DIR := $(NVHPC_ROOT)/math_libs/12.8/targets/x86_64-linux
INC        := -I$(CUDA_HOME)/include -I$(CUBLAS_DIR)/include -I.
LIBS       := -L$(CUDA_HOME)/lib64 -L$(CUBLAS_DIR)/lib -lcublas -lcudart

# ---------- Sources ----------
CPU_SRC    := cpu_baseline.cpp
GPU_SRCS   := main.cu gpu_cov.cu gpu_gradient.cu
GPU_HDRS   := utils.h gpu_cov.cuh gpu_gradient.cuh cpu_baseline_inline.h

# ---------- Outputs ----------
CPU_BIN    := cpu_baseline
GPU_BIN    := portfolio_gpu

# ---------------------------------------------------------------
.PHONY: all cpu gpu clean benchmark fetch_data run_cpu run_gpu

all: cpu gpu

cpu: $(CPU_BIN)

gpu: $(GPU_BIN)

# ---- CPU baseline (plain C++ — no CUDA libs needed)
$(CPU_BIN): $(CPU_SRC) utils.h
	$(CXX) $(CXX_FLAGS) -I. -o $@ $(CPU_SRC)

# ---- GPU binary (all .cu files together → single nvcc invocation)
$(GPU_BIN): $(GPU_SRCS) $(GPU_HDRS)
	$(NVCC) $(NVCC_FLAGS) $(INC) -o $@ $(GPU_SRCS) $(LIBS)

# ---- Fetch data
fetch_data:
	@echo "Fetching returns data..."
	pip install yfinance pandas --quiet
	python data/fetch_data.py --n 500 --years 5 --out data/returns.csv

# ---- Quick smoke-test runs
run_cpu: $(CPU_BIN)
	./$(CPU_BIN) data/returns.csv 1.0 2000 0.001

run_gpu: $(GPU_BIN)
	./$(GPU_BIN) data/returns.csv 1.0 2000 0.001

# ---- Full benchmark sweep
benchmark: all
	@echo "Running benchmark sweep..."
	mkdir -p plots
	python benchmark.py \
	    --csv     returns.csv \
	    --ns      50,100,150,200,250,300,350,400,500,750,1000,1250 \
	    --cpu_bin ./$(CPU_BIN) \
	    --gpu_bin ./$(GPU_BIN) \
	    --out_dir plots

# ---- Cleanup
clean:
	rm -f $(CPU_BIN) $(GPU_BIN) *.o *.d