#include "rdma_client.h"
#include <omp.h>

void client_start(struct rdma_client_t *res, const char *server_ip, uint32_t server_port, size_t mr_size, const char* dev_name) {

    int rc;

    rdma_client_t_init(res);
    if ((rc = rdma_client_t_create(res, server_ip, server_port, mr_size, dev_name) != 0)) {
        fprintf(stderr, "Create rdma_client_t failed\n");
        goto rdma_exit;
    }

    /* connect the QPs */
    if ((rc = connect_qp(res)) != 0) {
        fprintf(stderr, "failed to connect QPs\n");
        goto rdma_exit;
    }

	res->num_write = 0;
    res->num_read = 0;
    res->local_cache = NULL;

    memset(res->buffer, 0, mr_size);
    return;

rdma_exit:

    if ((rc = rdma_client_t_destroy(res)) != 0) {
        fprintf(stderr, "failed to destroy rdma_client_t\n");
    }
}

void poll_cq_batch(struct rdma_client_t* client, int num_wc)
{
	struct ibv_wc wc[num_wc];
	int ret = process_cq_event(client->cq, wc, num_wc);
	if (ret != num_wc)
			fprintf(stderr, "Failed to poll cq batch\n");
	// else printf("success\n");
	return;
}

int client_finalize(struct rdma_client_t* client, int fd, size_t file_size)
{
    char temp_char;
#ifdef WRITE_FILE
    for(unsigned long long int idx = SNAPSHOT_SIZE; idx < file_size; idx += MR_SIZE) {
        read_remote(client, 0, idx, MR_SIZE, 0);
        poll_cq_batch(client, 1);
        pwrite(fd, client->buffer, MR_SIZE, idx);
    }
    pwrite(fd, client->local_cache, SNAPSHOT_SIZE, 0);
#endif
	int ret = sock_sync_data(client->sock, 1, "Q", &temp_char);
    ret |= rdma_client_t_destroy(client);
    if (client->local_cache)
        free(client->local_cache);
	return ret;
}

int write_remote(struct rdma_client_t* client, uint64_t l_off, uint64_t r_off, uint32_t size, uint64_t id)
{
    return post_send(client, IBV_WR_RDMA_WRITE, size, l_off, r_off, id);
}

int read_remote(struct rdma_client_t* client, uint64_t l_off, uint64_t r_off, uint32_t size, uint64_t id)
{
    return post_send(client, IBV_WR_RDMA_READ, size, l_off, r_off, id);
}
