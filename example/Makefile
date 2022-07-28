SHELL=/bin/bash
.SUFFIXES:
default: stabilize.exe

OPENMP=yes
OPTIMIZE+=-march=x86-64
COMPILER=gcc-8
include $(ARCHLAB_ROOT)/cse141.make

stabilize.exe:  $(BUILD)main.o  $(BUILD)stabilize.o
matexp.exe: EXTRA_LDFLAGS=
$(BUILD)stabilize.o : OPTIMIZE=-O3
$(BUILD)stabilize.s : OPTIMIZE=-O3

DATASET=cifar

.PHONY: traces
traces:
	for i in baseline reorder_pixelxy tile_y_1 pretile_y; do  echo mtrace --trace $${i}_$(DATASET) --cache-line-count 128 -- ./stabilize.exe --dataset $(DATASET) --impl $$i;done

