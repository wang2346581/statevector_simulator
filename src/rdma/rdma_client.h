#include "rdma_common.h"

void client_start(struct rdma_client_t *res, const char *server_ip, uint32_t server_port, size_t mr_size, const char* dev_name);
void poll_cq_batch(struct rdma_client_t* client, int num_wc);
int client_finalize(struct rdma_client_t* client, int fd, size_t file_size);
int write_remote(struct rdma_client_t* client, uint64_t l_off, uint64_t r_off, uint32_t size, uint64_t id);
int read_remote(struct rdma_client_t* client, uint64_t l_off, uint64_t r_off, uint32_t size, uint64_t id);