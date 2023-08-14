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

#define MAX_CQ_CAPACITY 1024
#define MAX_SGE 2
#define MAX_WR 1024

#define DEFAULT_TCP_PORT 20886

#if __BYTE_ORDER == __LITTLE_ENDIAN
static inline uint64_t htonll(uint64_t x) { return bswap_64(x); }
static inline uint64_t ntohll(uint64_t x) { return bswap_64(x); }
#elif __BYTE_ORDER == __BIG_ENDIAN
static inline uint64_t htonll(uint64_t x) { return x; }
static inline uint64_t ntohll(uint64_t x) { return x; }
#else
#error __BYTE_ORDER is neither __LITTLE_ENDIAN nor __BIG_ENDIAN
#endif

/* structure of test parameters */
struct config_t
{
	const char *dev_name; /* IB device name */
	char *server_name;	/* server host name */
	u_int32_t tcp_port;   /* server TCP port */
	int ib_port;		  /* local IB port to work with */
	int gid_idx;		  /* gid index to use */
};

/* structure to exchange data which is needed to connect the QPs */
struct cm_con_data_t
{
	uint64_t addr;   /* Buffer address */
	uint32_t rkey;   /* Remote key */
	uint32_t qp_num; /* QP number */
	uint16_t lid;	/* LID of the IB port */
	uint8_t gid[16]; /* gid */
} __attribute__((packed));

/* structure of system resources */
struct resources
{
	struct ibv_device_attr device_attr; /* Device attributes */
	struct ibv_port_attr port_attr;	    /* IB port attributes */
	struct cm_con_data_t remote_props;  /* values to connect to remote side */
	struct ibv_context *ib_ctx;		    /* device handle */
	struct ibv_pd *pd;				    /* PD handle */
	struct ibv_cq *cq;				    /* CQ handle */
	struct ibv_qp *qp;				    /* QP handle */
	struct ibv_mr *mr;				    /* MR handle for buf */
	void *buf;						    /* memory buffer pointer, used for RDMA and send ops */
	int sock;						    /* TCP socket file descriptor */
};

void print_config(struct config_t *config);
void resources_init(struct resources *res);
int resources_create(struct resources *res, uint32_t server_port, size_t mr_size, const char* dev_name);
int resources_create_exit(struct resources *res, struct ibv_device **dev_list, int rc);
int resources_destroy(struct resources *res);
int connect_qp(struct resources *res);
int poll_completion(struct resources *res);
int sock_sync_data(int sock, int xfer_size, char *local_data, char *remote_data);

/* TODO */
/* 
 * Still Need:
 *       1. uint64_t l_offset
 *       2. uint64_t r_offset
 *       3. uint32_t size
 *       4. uint64_t id                       */
int post_send(struct resources *res, int opcode, uint32_t size, uint64_t l_offset, uint64_t r_offset);
