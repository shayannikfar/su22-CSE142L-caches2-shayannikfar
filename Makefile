SHELL=/bin/bash
.SUFFIXES:
default:

.PHONY: create-labs
create-labs:
	cse142 lab delete -f caches2-bench
	cse142 lab delete -f caches2
	cse142 lab create --name "Lab 4: Caches II (Benchmark)" --short-name "caches2-bench" --docker-image stevenjswanson/cse142l-runner:latest --execution-time-limit 0:05:00 --total-time-limit 1:00:00 --due-date 2021-11-24T23:59:59 --starter-repo https://github.com/CSE142/fa21-CSE142L-caches2-starter.git --starter-branch main
	cse142 lab create --name "Lab 4: Caches II" --short-name "caches2" --docker-image stevenjswanson/cse142l-runner:v47 --execution-time-limit 0:05:00 --total-time-limit 1:00:00 --due-date 2021-11-24T23:59:59

STUDENT_EDITABLE_FILES=matexp_solution.hpp config.make
PRIVATE_FILES=Lab.key.ipynb admin .git solution bad-solution

OPTIMIZE+=-march=x86-64
COMPILER=gcc-8
include $(ARCHLAB_ROOT)/cse141.make

.PHONY: autograde
autograde: matexp.exe regressions.json bench.csv

bench.csv:
	./matexp.exe --MHz 3500 --stats bench.csv --stat-set ./L1.cfg --function bench_solution

#run_tests.exe: $(BUILD)ChunkAlloc.o
regressions.json: run_tests.exe
	./run_tests.exe --gtest_output=json:$@
test: regressions.json

matexp.exe:  $(BUILD)matexp_main.o  $(BUILD)matexp.o
matexp.exe: EXTRA_LDFLAGS=
$(BUILD)matexp.o : OPTIMIZE=$(MATEXP_OPTIMIZE)
$(BUILD)matexp.s : OPTIMIZE=$(MATEXP_OPTIMIZE)
$(BUILD)matexp_main.o : OPTIMIZE=$(MATEXP_OPTIMIZE)

$(BUILD)run_tests.o : OPTIMIZE=-O3
fiddle.exe:  $(BUILD)fiddle.o $(FIDDLE_OBJS)
fiddle.exe: EXTRA_LDFLAGS=-pg
$(BUILD)fiddle.o : OPTIMIZE=-O3 -pg


-include $(DJR_JOB_ROOT)/$(LAB_SUBMISSION_DIR)/config.make
