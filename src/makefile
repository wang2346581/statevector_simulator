all: qSim.out
CC:=gcc
INCLUDE:=.
CFLAGS:=-g -O3 -lrdmacm -libverbs -march=native
OMPFLAGES:=-fopenmp
MEMORY_FLAGS:=-D MEMORY

RDMA_FILES = rdma/rdma_client.c rdma/rdma_common.c
RDMA_OBJS = $(RDMA_FILES:.c=.o)

qSim.out: main.o ini.o init.o common.o gate.o gate_util.o gate_chunk.o measure.o $(RDMA_OBJS)
	$(CC) $^ -o $@ -lm $(CFLAGS) $(OMPFLAGES) $(MEMORY_FLAGS)
# -lm for math.h

%.o: %.c
	$(CC) -c $^ -o $@ $(CFLAGS) $(OMPFLAGES) $(MEMORY_FLAGS)

clean:
	rm -f qSim.out *.o $(RDMA_OBJS)

run: all
	make clean
	make
	./qSim.out -c circuit/8.txt -i myIni.ini
