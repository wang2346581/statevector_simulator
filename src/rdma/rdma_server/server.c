#include "server.h"

void server_start(struct resources *res, uint32_t server_port, size_t mr_size, const char* dev_name) {

    int rc;

    resources_init(res);
    if ((rc = resources_create(res, server_port, mr_size, dev_name) != 0)) {
        fprintf(stderr, "Create resources failed\n");
        goto rdma_exit;
    }

    memset(res->buf, 0, mr_size);

    /* connect the QPs */
    if ((rc = connect_qp(res)) != 0) {
        fprintf(stderr, "failed to connect QPs\n");
        goto rdma_exit;
    }

    return;

rdma_exit:

    if ((rc = resources_destroy(res)) != 0) {
        fprintf(stderr, "failed to destroy resources\n");
    }
}

int main(int argc, char *argv[]) {
    char temp_char;
    struct resources *res;

retry:
    res = malloc (sizeof(struct resources) * 16);
    server_start(res, atoll(argv[1]), SERVER_MSG_SIZE, NULL);
    sock_sync_data(res->sock, 1, "Q", &temp_char);
    sock_sync_data(res->sock, 1, "Q", &temp_char);
    resources_destroy(res);
    free(res);
    printf("Finish connection at port %s\n", argv[1]);

    sleep(1);
    goto retry;

    return 0;
}
