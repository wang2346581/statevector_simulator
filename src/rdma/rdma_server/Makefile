COMPILER = gcc
CFLAGS = -lrdmacm -libverbs -pthread -g -Wall -O3
# DEBUGFLAGS = -D DEBUG

SERVER_FILE = server.c util.c

SERVER_OBJ = $(SERVER_FILE:.c=.o)

all: rdma_server

rdma_server: $(SERVER_OBJ)
	$(COMPILER) $^ -o $@ $(CFLAGS) $(DEBUGFLAGS)

%.o: %.c
	$(COMPILER) -c $^ -o $@ $(CFLAGS) $(DEBUGFLAGS)

clean:
	rm -f server
	rm -f $(SERVER_OBJ)
