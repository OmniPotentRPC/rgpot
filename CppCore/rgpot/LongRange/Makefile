# Reproduction for the long-range summands.
# linkcell is built first. s-dftd3, dftd4 and nvcc are optional.
#
#   make -C CppCore/rgpot/LongRange test
#   python3 tools/lr_physics/validate.py /tmp/lr_phys_test
#   make -C CppCore/rgpot/LongRange bench

ROOT := ../../..
LC ?= /tmp/linkcell-src
CXX ?= g++
NVCC ?= nvcc
CXXFLAGS := -std=c++20 -O2 -g -fopenmp -Wall -Wextra
CXXFLAGS += -I$(ROOT)/CppCore -I$(ROOT)/include -I$(LC)/include
CXXFLAGS += -I/tmp -DRGPOT_HAS_DFTD3 -DRGPOT_HAS_DFTD4
LIBS := -Wl,--whole-archive $(LC)/target/release/liblinkcell.a -Wl,--no-whole-archive
LIBS += -pthread -ldl -lm -lcudart
DFTD3_SO ?= /home/ubuntu/.local/lib/python3.12/site-packages/dftd3/_libdftd3.cpython-312-x86_64-linux-gnu.so
DFTD4_SO ?= /home/ubuntu/.local/lib/python3.12/site-packages/dftd4/_libdftd4.cpython-312-x86_64-linux-gnu.so
PYLIB := -L/usr/lib/python3.12/config-3.12-x86_64-linux-gnu -lpython3.12

SRCS := Pairs.cc Electro.cc Dispersion.cc Nlh.cc SumPot.cc CudaPath.cc
OBJS := $(SRCS:.cc=.o)

.PHONY: test bench sets cuda clean linkcell

linkcell:
	cargo build --release --manifest-path $(LC)/Cargo.toml

CudaPath.cu.o: CudaPath.cu CudaPath.hpp CudaMath.cuh
	$(NVCC) -arch=sm_70 -std=c++20 -c -I$(ROOT)/CppCore -I$(ROOT)/include CudaPath.cu -o CudaPath.cu.o

test: $(OBJS) CudaPath.cu.o
	$(CXX) $(CXXFLAGS) -o /tmp/lr_phys_test $(OBJS) CudaPath.cu.o $(ROOT)/CppCore/tests/lr_phys_test.cc $(DFTD3_SO) $(DFTD4_SO) $(LIBS) $(PYLIB)
	/tmp/lr_phys_test

bench: $(OBJS)
	$(CXX) $(CXXFLAGS) -o /tmp/lr_phys_bench $(OBJS) $(ROOT)/CppCore/tests/lr_phys_bench.cc $(DFTD3_SO) $(DFTD4_SO) $(LIBS) $(PYLIB)
	/tmp/lr_phys_bench 6

sets: $(OBJS)
	$(CXX) $(CXXFLAGS) -o /tmp/lr_sets $(OBJS) $(ROOT)/CppCore/tests/lr_sets.cc $(DFTD3_SO) $(DFTD4_SO) $(LIBS) $(PYLIB)

%.o: %.cc
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) CudaPath.cu.o
