# Reproduction for the long-range summands.
# linkcell is built first; s-dftd3 and dftd4 are optional.
#
#   make -C CppCore/rgpot/LongRange test
#   python3 tools/lr_physics/validate.py /tmp/lr_phys_test
#   make -C CppCore/rgpot/LongRange bench

ROOT := ../../..
LC ?= /tmp/linkcell-src
CXX ?= g++
CXXFLAGS := -std=c++20 -O2 -g -fopenmp -Wall -Wextra
CXXFLAGS += -I$(ROOT)/CppCore -I$(ROOT)/include -I$(LC)/include
LIBS := -Wl,--whole-archive $(LC)/target/release/liblinkcell.a -Wl,--no-whole-archive
LIBS += -pthread -ldl -lm

SRCS := Pairs.cc Electro.cc Dispersion.cc Nlh.cc SumPot.cc
OBJS := $(SRCS:.cc=.o)

.PHONY: test bench clean linkcell

linkcell:
	cargo build --release --manifest-path $(LC)/Cargo.toml

test: $(OBJS)
	$(CXX) $(CXXFLAGS) -o /tmp/lr_phys_test $(OBJS) $(ROOT)/CppCore/tests/lr_phys_test.cc $(LIBS)
	/tmp/lr_phys_test

bench: $(OBJS)
	$(CXX) $(CXXFLAGS) -o /tmp/lr_phys_bench $(OBJS) $(ROOT)/CppCore/tests/lr_phys_bench.cc $(LIBS)
	/tmp/lr_phys_bench 6

%.o: %.cc
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS)
