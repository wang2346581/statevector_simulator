#include "rdma_common.h"

unsigned long long int SNAPSHOT_SIZE; // snapshot size for each thread
unsigned long long int MR_SIZE; // buffer_size for each thread

void rdma_client_t_init(struct rdma_client_t *res) {
	memset(res, 0, sizeof *res);
	res->sock = -1;
}

int sock_connect(const char *servername, int port)
{
	struct addrinfo *resolved_addr = NULL;
	struct addrinfo *iterator;
	char service[6];
	int sockfd = -1;
	int listenfd = 0;
	int tmp;
	struct addrinfo hints = {
		.ai_flags = AI_PASSIVE,
		.ai_family = AF_INET,
		.ai_socktype = SOCK_STREAM
	};

	if (sprintf(service, "%d", port) < 0)
		goto sock_connect_exit;

	/* Resolve DNS address, use sockfd as temp storage */
	sockfd = getaddrinfo(servername, service, &hints, &resolved_addr);

	if (sockfd < 0) {
		fprintf(stderr, "%s for %s:%d\n", gai_strerror(sockfd), servername, port);
		goto sock_connect_exit;
	}

	/* Search through results and find the one we want */
	for (iterator = resolved_addr; iterator; iterator = iterator->ai_next)
	{
		sockfd = socket(iterator->ai_family, iterator->ai_socktype, iterator->ai_protocol);
		if (sockfd >= 0) {
			if (servername) {
				/* Client mode. Initiate connection to remote */
				if ((tmp = connect(sockfd, iterator->ai_addr, iterator->ai_addrlen)))
				{
					fprintf(stderr, "failed connect \n");
					close(sockfd);
					sockfd = -1;
				}
            } else {
					/* Server mode. Set up listening socket an accept a connection */
					listenfd = sockfd;
					sockfd = -1;
					if (bind(listenfd, iterator->ai_addr, iterator->ai_addrlen))
						goto sock_connect_exit;
					listen(listenfd, 1);
					sockfd = accept(listenfd, NULL, 0);
			}
		}
	}

sock_connect_exit:
	if (listenfd)
		close(listenfd);
	if (resolved_addr)
		freeaddrinfo(resolved_addr);
	if (sockfd < 0) {
		if (servername)
			fprintf(stderr, "Couldn't connect to %s:%d\n", servername, port);
		else {
			perror("server accept");
			fprintf(stderr, "accept() failed\n");
		}
	}
	return sockfd;
}

int rdma_client_t_create_exit(struct rdma_client_t *res, struct ibv_device **dev_list, int rc) {
	if (res->sock >= 0) {
        if (close(res->sock))
            fprintf(stderr, "failed to close socket\n");
        res->sock = -1;
    }
	if (dev_list) {
		ibv_free_device_list(dev_list);
		dev_list = NULL;
	}
	if (res->ib_ctx) {
		ibv_close_device(res->ib_ctx);
		res->ib_ctx = NULL;
	}
	if (res->pd) {
		ibv_dealloc_pd(res->pd);
		res->pd = NULL;
	}
	if (res->cq) {
		ibv_destroy_cq(res->cq);
		res->cq = NULL;
	}
	if (res->buffer) {
		free(res->buffer);
		res->buffer = NULL;
	}
	if (res->mr) {
		ibv_dereg_mr(res->mr);
		res->mr = NULL;
	}
	if (res->qp) {
		ibv_destroy_qp(res->qp);
		res->qp = NULL;
	}
    return rc;
}

int rdma_client_t_create(struct rdma_client_t *res, const char *server_ip, uint32_t server_port, size_t mr_size, const char* dev_name) {
    struct ibv_device **dev_list = NULL;
	struct ibv_qp_init_attr qp_init_attr;
	struct ibv_device *ib_dev = NULL;
	size_t size;
	int i;
	int mr_flags = 0;
	int cq_size = 0;
	int num_devices;

	res->sock = sock_connect(server_ip, server_port);
	if (res->sock < 0) {
		fprintf(stderr, "failed to establish TCP connection to server %s, port %d\n",
				server_ip, server_port);
		return rdma_client_t_create_exit(res, dev_list, -1);
	}

#ifdef DEBUG
	fprintf(stdout, "TCP connection was established\n");
	fprintf(stdout, "searching for IB devices in host\n");
#endif

	/* get device names in the system */
	dev_list = ibv_get_device_list(&num_devices);
	if (!dev_list) {
		fprintf(stderr, "failed to get IB devices list\n");
		return rdma_client_t_create_exit(res, dev_list, 1);
	}

	/* if there isn't any IB device in host */
	if (!num_devices) {
		fprintf(stderr, "found %d device(s)\n", num_devices);
		return rdma_client_t_create_exit(res, dev_list, 1);
	}

#ifdef DEBUG
	fprintf(stdout, "found %d device(s)\n", num_devices);
#endif

	/* search for the specific device we want to work with */
	for (i = 0; i < num_devices; i++)
	{
		if (!dev_name)
		{
			dev_name = strdup(ibv_get_device_name(dev_list[num_devices - 1]));
#ifdef DEBUG
			fprintf(stdout, "device not specified, using last one found: %s\n", dev_name);
#endif
		}
		if (!strcmp(ibv_get_device_name(dev_list[i]), dev_name))
		{
			ib_dev = dev_list[i];
			break;
		}
	}

	/* if the device wasn't found in host */
	if (!ib_dev) {
		fprintf(stderr, "IB device %s wasn't found\n", dev_name);
		return rdma_client_t_create_exit(res, dev_list, 1);
	}

	/* get device handle */
	res->ib_ctx = ibv_open_device(ib_dev);
	if (!res->ib_ctx) {
		fprintf(stderr, "failed to open device %s\n", dev_name);
		return rdma_client_t_create_exit(res, dev_list, 1);
	}

#ifdef DEBUG
	fprintf(stdout, "found device %s\n", dev_name);
#endif

	/* We are now done with device list, free it */
	ibv_free_device_list(dev_list);
	dev_list = NULL;
	ib_dev = NULL;

	/* query port properties */
	if (ibv_query_port(res->ib_ctx, 1, &res->port_attr)) {
		fprintf(stderr, "ibv_query_port on port %u failed\n", 1);
		return rdma_client_t_create_exit(res, dev_list, 1);
	}

	/* allocate Protection Domain */
	res->pd = ibv_alloc_pd(res->ib_ctx);
	if (!res->pd) {
		fprintf(stderr, "ibv_alloc_pd failed\n");
		return rdma_client_t_create_exit(res, dev_list, 1);
	}

	/* each side will send only one WR, so Completion Queue with 1 entry is enough */
	cq_size = MAX_CQ_CAPACITY;
	res->cq = ibv_create_cq(res->ib_ctx, cq_size, NULL, NULL, 0);
	if (!res->cq) {
		fprintf(stderr, "failed to create CQ with %u entries\n", cq_size);
		return rdma_client_t_create_exit(res, dev_list, 1);
	}


	/* allocate the memory buffer that will hold the data */
	size = mr_size;
#ifdef DEBUG
	printf("Size of memory region is %ld\n", size);
#endif
	res->buffer = malloc(size);
	if (!res->buffer) {
		fprintf(stderr, "failed to malloc %Zu bytes to memory buffer\n", size);
		return rdma_client_t_create_exit(res, dev_list, 1);
	}
	memset(res->buffer, 0, size);

	/* register the memory buffer */
	mr_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE;
	res->mr = ibv_reg_mr(res->pd, res->buffer, size, mr_flags);
	if (!res->mr) {
		fprintf(stderr, "ibv_reg_mr failed with mr_flags=0x%x\n", mr_flags);
		return rdma_client_t_create_exit(res, dev_list, 1);
	}

#ifdef DEBUG
	fprintf(stdout, "MR was registered with addr=%p, lkey=0x%x, rkey=0x%x, flags=0x%x\n",
			res->buffer, res->mr->lkey, res->mr->rkey, mr_flags);
#endif

	/* create the Queue Pair */
	memset(&qp_init_attr, 0, sizeof(qp_init_attr));
	qp_init_attr.qp_type = IBV_QPT_RC;
	// qp_init_attr.sq_sig_all = 1;
	qp_init_attr.send_cq = res->cq;
	qp_init_attr.recv_cq = res->cq;
	qp_init_attr.cap.max_send_wr = MAX_WR;
	qp_init_attr.cap.max_recv_wr = MAX_WR;
	qp_init_attr.cap.max_send_sge = MAX_SGE;
	qp_init_attr.cap.max_recv_sge = MAX_SGE;
	res->qp = ibv_create_qp(res->pd, &qp_init_attr);
	if (!res->qp) {
		fprintf(stderr, "failed to create QP\n");
		return rdma_client_t_create_exit(res, dev_list, 1);
	}

#ifdef DEBUG
	fprintf(stdout, "QP was created, QP number=0x%x\n", res->qp->qp_num);
#endif

	return 0;
}

int poll_completion(struct rdma_client_t *res) {
	struct ibv_wc wc;
	unsigned long start_time_msec;
	unsigned long cur_time_msec;
	struct timeval cur_time;
	int poll_result;
	int rc = 0;

	/* poll the completion for a while before giving up of doing it .. */
	gettimeofday(&cur_time, NULL);
	start_time_msec = (cur_time.tv_sec * 1000) + (cur_time.tv_usec / 1000);
	
	do {
		poll_result = ibv_poll_cq(res->cq, 1, &wc);
		gettimeofday(&cur_time, NULL);
		cur_time_msec = (cur_time.tv_sec * 1000) + (cur_time.tv_usec / 1000);
	} while ((poll_result == 0) && ((cur_time_msec - start_time_msec) < MAX_POLL_CQ_TIMEOUT));
	
	if (poll_result < 0) {
		/* poll CQ failed */
		fprintf(stderr, "poll CQ failed\n");
		rc = 1;
	} else if (poll_result == 0) { /* the CQ is empty */
		fprintf(stderr, "completion wasn't found in the CQ after timeout\n");
		rc = 1;
	} else {
		/* CQE found */
#ifdef DEBUG
		fprintf(stdout, "completion was found in CQ with status 0x%x\n", wc.status);
#endif
		/* check the completion status (here we don't care about the completion opcode */
		if (wc.status != IBV_WC_SUCCESS) {
			fprintf(stderr, "got bad completion with status: 0x%x, vendor syndrome: 0x%x\n", wc.status,
					wc.vendor_err);
			rc = 1;
		}
	}

	return rc;
}

int rdma_client_t_destroy(struct rdma_client_t *res)
{
	int rc = 0;
	if (res->qp)
		if (ibv_destroy_qp(res->qp))
		{
			fprintf(stderr, "failed to destroy QP\n");
			rc = 1;
		}
	if (res->mr)
		if (ibv_dereg_mr(res->mr))
		{
			fprintf(stderr, "failed to deregister MR\n");
			rc = 1;
		}
	if (res->buffer)
		free(res->buffer);
	if (res->cq)
		if (ibv_destroy_cq(res->cq))
		{
			fprintf(stderr, "failed to destroy CQ\n");
			rc = 1;
		}
	if (res->pd)
		if (ibv_dealloc_pd(res->pd))
		{
			fprintf(stderr, "failed to deallocate PD\n");
			rc = 1;
		}
	if (res->ib_ctx)
		if (ibv_close_device(res->ib_ctx))
		{
			fprintf(stderr, "failed to close device context\n");
			rc = 1;
		}
	if (res->sock >= 0)
		if (close(res->sock))
		{
			fprintf(stderr, "failed to close socket\n");
			rc = 1;
		}
	return rc;
}

int sock_sync_data(int sock, int xfer_size, char *local_data, char *remote_data) {
	int rc;
	int read_bytes = 0;
	int total_read_bytes = 0;

	if ((rc = write(sock, local_data, xfer_size)) < xfer_size)
		fprintf(stderr, "Failed writing data during sock_sync_data\n");
	else
		rc = 0;

	while (!rc && total_read_bytes < xfer_size) {
		read_bytes = read(sock, remote_data, xfer_size);
		if (read_bytes > 0)
			total_read_bytes += read_bytes;
		else
			rc = read_bytes;
	}
	return rc;
}

int post_receive(struct rdma_client_t *res, uint32_t size, uint64_t l_offset) {

	struct ibv_recv_wr rr;
	struct ibv_sge sge;
	struct ibv_recv_wr *bad_wr;
	int rc;

	/* prepare the scatter/gather entry */
	memset(&sge, 0, sizeof(sge));
	sge.addr = (uint64_t)res->buffer + l_offset;
	sge.length = size;
	sge.lkey = res->mr->lkey;
	/* prepare the receive work request */

	memset(&rr, 0, sizeof(rr));
	rr.next = NULL;
	rr.wr_id = 0;
	rr.sg_list = &sge;
	rr.num_sge = 1;

	/* post the Receive Request to the RQ */
	rc = ibv_post_recv(res->qp, &rr, &bad_wr);

	if ((rc = ibv_post_recv(res->qp, &rr, &bad_wr)) != 0)
		fprintf(stderr, "failed to post RR\n");
	else {

#ifdef DEBUG
	fprintf(stdout, "Receive Request was posted\n");
#endif

	}

	return rc;
}

int modify_qp_to_init(struct ibv_qp *qp) {
	struct ibv_qp_attr attr;
	int flags;
	int rc;

	memset(&attr, 0, sizeof(attr));
	attr.qp_state = IBV_QPS_INIT;
	attr.port_num = 1;
	attr.pkey_index = 0;
	attr.qp_access_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE;
	flags = IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS;

	if ((rc = ibv_modify_qp(qp, &attr, flags)) != 0)
		fprintf(stderr, "failed to modify QP state to INIT\n");

	return rc;
}

/* make the status of QP ready to receive */
int modify_qp_to_rtr(struct ibv_qp *qp, uint32_t remote_qpn, uint16_t dlid, uint8_t *dgid) {
	struct ibv_qp_attr attr;
	int flags;
	int rc;

	memset(&attr, 0, sizeof(attr));
	attr.qp_state = IBV_QPS_RTR;
	attr.path_mtu = IBV_MTU_256;
	attr.dest_qp_num = remote_qpn;
	attr.rq_psn = 0;
	attr.max_dest_rd_atomic = 1;
	attr.min_rnr_timer = 0x12;
	attr.ah_attr.is_global = 0;
	attr.ah_attr.dlid = dlid;
	attr.ah_attr.sl = 0;
	attr.ah_attr.src_path_bits = 0;
	attr.ah_attr.port_num = 1;
	attr.ah_attr.is_global = 1;
	attr.ah_attr.port_num = 1;
	memcpy(&attr.ah_attr.grh.dgid, dgid, 16);
	attr.ah_attr.grh.flow_label = 0;
	attr.ah_attr.grh.hop_limit = 1;
	attr.ah_attr.grh.sgid_index = 0;
	attr.ah_attr.grh.traffic_class = 0;

	flags = IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN |
			IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER;
	
	if ((rc = ibv_modify_qp(qp, &attr, flags)) != 0)
		fprintf(stderr, "failed to modify QP state to RTR\n");
	
	return rc;
}

/* make the status of QP ready to send */
int modify_qp_to_rts(struct ibv_qp *qp) {
	struct ibv_qp_attr attr;
	int flags;
	int rc;

	memset(&attr, 0, sizeof(attr));
	attr.qp_state = IBV_QPS_RTS;
	attr.timeout = 0x12;
	attr.retry_cnt = 6;
	attr.rnr_retry = 0;
	attr.sq_psn = 0;
	attr.max_rd_atomic = 1;

	flags = IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT |
			IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC;
	
	if ((rc = ibv_modify_qp(qp, &attr, flags)) != 0)
		fprintf(stderr, "failed to modify QP state to RTS\n");
	
	return rc;
}

int connect_qp(struct rdma_client_t *res) {
	struct cm_con_data_t local_con_data;
	struct cm_con_data_t remote_con_data;
	struct cm_con_data_t tmp_con_data;
	int rc = 0;
	char temp_char;
	union ibv_gid my_gid;

	if ((rc = ibv_query_gid(res->ib_ctx, 1, 0, &my_gid)) != 0) {
		fprintf(stderr, "could not get gid for port %d, index %d\n", 1, 0);
		return rc;
	}

	/* exchange using TCP sockets info required to connect QPs */
	local_con_data.addr = htonll((uintptr_t)res->buffer);
	local_con_data.rkey = htonl(res->mr->rkey);
	local_con_data.qp_num = htonl(res->qp->qp_num);
	local_con_data.lid = htons(res->port_attr.lid);
	memcpy(local_con_data.gid, &my_gid, 16);

#ifdef DEBUG
	fprintf(stdout, "\nLocal LID = 0x%x\n", res->port_attr.lid);
#endif

	if (sock_sync_data(res->sock, sizeof(struct cm_con_data_t), (char *)&local_con_data, (char *)&tmp_con_data) < 0) {
		fprintf(stderr, "failed to exchange connection data between sides\n");
		return rc;
	}

	remote_con_data.addr = ntohll(tmp_con_data.addr);
	remote_con_data.rkey = ntohl(tmp_con_data.rkey);
	remote_con_data.qp_num = ntohl(tmp_con_data.qp_num);
	remote_con_data.lid = ntohs(tmp_con_data.lid);
	memcpy(remote_con_data.gid, tmp_con_data.gid, 16);

	/* save the remote side attributes, we will need it for the post SR */
	res->remote_props = remote_con_data;

#ifdef DEBUG
	fprintf(stdout, "Remote address = 0x%" PRIx64 "\n", remote_con_data.addr);
	fprintf(stdout, "Remote rkey = 0x%x\n", remote_con_data.rkey);
	fprintf(stdout, "Remote QP number = 0x%x\n", remote_con_data.qp_num);
	fprintf(stdout, "Remote LID = 0x%x\n", remote_con_data.lid);
	uint8_t *p = remote_con_data.gid;
	fprintf(stdout, "Remote GID =%02x:%02x:%02x:%02x:%02x:%02x:%02x:%02x:%02x:%02x:%02x:%02x:%02x:%02x:%02x:%02x\n", p[0],
					 p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9], p[10], p[11], p[12], p[13], p[14], p[15]);
#endif

	/* modify the QP to init */
	if((modify_qp_to_init(res->qp)) != 0) {
		fprintf(stderr, "change QP state to INIT failed\n");
		return rc;
	}

	if ((rc = modify_qp_to_rtr(res->qp, remote_con_data.qp_num, remote_con_data.lid, remote_con_data.gid)) != 0) {
		fprintf(stderr, "failed to modify QP state to RTR\n");
		return rc;
	}

	if((rc = modify_qp_to_rts(res->qp)) != 0) {
		fprintf(stderr, "failed to modify QP state to RTR\n");
		return rc;
	}

#ifdef DEBUG
	fprintf(stdout, "QP state was change to RTS\n");
#endif

	/* just send a dummy char back and forth */ 
	if (sock_sync_data(res->sock, 1, "Q", &temp_char)) {
		fprintf(stderr, "sync error after QPs are were moved to RTS\n");
		return 1;
	}

	return rc;
}

/* TODO */
/* 
 * Still Need:
 *       1. uint64_t l_offset
 *       2. uint64_t r_offset
 *       3. uint32_t size
 *       4. uint64_t id                       */
int post_send(struct rdma_client_t *res, int opcode, uint32_t size, 
							uint64_t l_offset, uint64_t r_offset, uint64_t id) {

	struct ibv_send_wr sr;
	struct ibv_sge sge;
	struct ibv_send_wr *bad_wr = NULL;
	int rc;

	/* prepare the scatter/gather entry */
	memset(&sge, 0, sizeof(sge));
	sge.addr = (uint64_t)res->buffer + l_offset;
	sge.length = size;
	sge.lkey = res->mr->lkey;

#ifdef DEBUG
	printf("RDMA Operation: l_offset = %ld, size = %d\n", l_offset, size);
#endif

	/* prepare the send work request */
	memset(&sr, 0, sizeof(sr));
	sr.next = NULL;
	sr.wr_id = id;
	sr.sg_list = &sge;
	sr.num_sge = 1;
	sr.opcode = opcode;
	sr.send_flags = IBV_SEND_SIGNALED;

	if (opcode != IBV_WR_SEND) {
		sr.wr.rdma.remote_addr = res->remote_props.addr + r_offset;
		sr.wr.rdma.rkey = res->remote_props.rkey;
	}

	/* there is a Receive Request in the responder side, so we won't get any into RNR flow */
	if ((rc = ibv_post_send(res->qp, &sr, &bad_wr)) != 0)
		fprintf(stderr, "failed to post SR\n");
	else {
#ifdef DEBUG
		switch (opcode) {
			case IBV_WR_SEND:
				fprintf(stdout, "Send Request was posted\n");
				break;
			case IBV_WR_RDMA_READ:
				fprintf(stdout, "RDMA Read Request was posted\n");
				break;
			case IBV_WR_RDMA_WRITE:
				fprintf(stdout, "RDMA Write Request was posted\n");
				break;
			default:
				fprintf(stdout, "Unknown Request was posted\n");
				break;
		}
#endif
	}

	return rc;
}

int process_cq_event(struct ibv_cq *cq, struct ibv_wc *wc, int max_wc)
{
	int ret = -1, i, total_wc = 0;

	total_wc = 0;
	do {
		ret = ibv_poll_cq(cq /* the CQ, we got notification for */,
						  max_wc - total_wc /* number of remaining WC elements*/,
						  wc + total_wc /* where to store */);
		if (ret < 0) {
			fprintf(stderr, "Failed to poll cq for wc due to %d \n", ret);
			return ret; /* ret is errno here */
		}
		total_wc += ret;
	} while (total_wc < max_wc);
#ifdef DEBUG
	printf("%d WC are completed \n", total_wc);
#endif

	/* Now we check validity and status of I/O work completions */
	for (i = 0; i < total_wc; i++) {
		if (wc[i].status != IBV_WC_SUCCESS) {
			fprintf(stdout, "Work completion (WC) has error status: %s at index %d\n",
					   ibv_wc_status_str(wc[i].status), i);
			/* return negative value */
			return -(wc[i].status);
		}
	}

	/* Return the number of wc we get */
	return total_wc;
}