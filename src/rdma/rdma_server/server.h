#include "util.h"

#define SERVER_MSG_SIZE (16L << 30)

void server_start(struct resources *res, uint32_t server_port, size_t mr_size, const char* dev_name);
