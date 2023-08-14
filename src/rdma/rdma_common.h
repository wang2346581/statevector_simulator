#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <stdint.h>
#include <inttypes.h>
#include <endian.h>
#include <byteswap.h>
#include <getopt.h>

#include <sys/time.h>
#include <arpa/inet.h>
#include <infiniband/verbs.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netdb.h>

extern unsigned long long int SNAPSHOT_SIZE; // snapshot size for each thread
extern unsigned long long int MR_SIZE; // buffer_size for each thread

/* poll CQ timeout in millisec (2 seconds) */
#define MAX_POLL_CQ_TIMEOUT 2000
#define MAX_CQ_CAPACITY 1024
#define MAX_SGE 2
#define MAX_WR 1024

#if __BYTE_ORDER == __LITTLE_ENDIAN
static inline uint64_t htonll(uint64_t x) { return bswap_64(x); }
static inline uint64_t ntohll(uint64_t x) { return bswap_64(x); }
#elif __BYTE_ORDER == __BIG_ENDIAN
static inline uint64_t htonll(uint64_t x) { return x; }
static inline uint64_t ntohll(uint64_t x) { return x; }
#else
#error __BYTE_ORDER is neither __LITTLE_ENDIAN nor __BIG_ENDIAN
#endif

/* structure to exchange data which is needed to connect the QPs */
struct cm_con_data_t
{
	uint64_t addr;   /* Buffer address */
	uint32_t rkey;   /* Remote key */
	uint32_t qp_num; /* QP number */
	uint16_t lid;	/* LID of the IB port */
	uint8_t gid[16]; /* gid */
} __attribute__((packed));

/* structure of system rdma_client_t */
struct rdma_client_t
{
	struct ibv_device_attr device_attr; /* Device attributes */
	struct ibv_port_attr port_attr;	    /* IB port attributes */
	struct cm_con_data_t remote_props;  /* values to connect to remote side */
	struct ibv_context *ib_ctx;		    /* device handle */
	struct ibv_pd *pd;				    /* PD handle */
	struct ibv_cq *cq;				    /* CQ handle */
	struct ibv_qp *qp;				    /* QP handle */
	struct ibv_mr *mr;				    /* MR handle for buf */
	void *buffer;						    /* memory buffer pointer, used for RDMA and send ops */
	int sock;						    /* TCP socket file descriptor */
	int num_write;
	int num_read;
	void *local_cache;
};

struct rdma_connection
{
	struct rdma_client_t res;
};

void rdma_client_t_init(struct rdma_client_t *res);
int rdma_client_t_create(struct rdma_client_t *res, const char *server_ip, uint32_t server_port, size_t mr_size, const char* dev_name);
int rdma_client_t_create_exit(struct rdma_client_t *res, struct ibv_device **dev_list, int rc);
int rdma_client_t_destroy(struct rdma_client_t *res);
int connect_qp(struct rdma_client_t *res);
int poll_completion(struct rdma_client_t *res);
int sock_sync_data(int sock, int xfer_size, char *local_data, char *remote_data);
int process_cq_event(struct ibv_cq *cq, struct ibv_wc *wc, int max_wc);

/* TODO */
/* 
 * Still Need:
 *       1. uint64_t l_offset
 *       2. uint64_t r_offset
 *       3. uint32_t size
 *       4. uint64_t id                       */
int post_send(struct rdma_client_t *res, int opcode, uint32_t size, uint64_t l_offset, uint64_t r_offset, uint64_t id);
