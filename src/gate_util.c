#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include "common.h"
#include "gate.h"
#include "gate_util.h"
#include <omp.h>

#define min(a,b) ((a) < (b) ? (a) : (b))

void set_outer(ull outer){
    _outer = outer;
    _half_outer = _outer/2;
    _half_outer_size = _half_outer * sizeof(Type);
}

inline void _thread_CX(setStreamv2 *s, int *counter){
    for(ull i = 0; i < loop_size; i += _outer){
        inner_loop_func(_outer, s->rd, s->fd, s->fd_index, s->fd_off, counter);
    }
}

inline void _thread_CX2(setStreamv2 *s, int *counter){
    for(ull i = 0; i < loop_size; i += _outer){
        inner_loop_func(_half_outer, s->rd, s->fd, s->fd_index, s->fd_off, counter);
        s->fd_off[0] += _half_outer_size;
        s->fd_off[1] += _half_outer_size;
    }
}

inline void _thread_CX4(setStreamv2 *s, int *counter){
    for(ull i = 0; i < loop_size; i += _outer){
        inner_loop_func(_half_outer, s->rd, s->fd, s->fd_index, s->fd_off, counter);
        s->fd_off[0] += _half_outer_size;
        s->fd_off[1] += _half_outer_size;
        s->fd_off[2] += _half_outer_size;
        s->fd_off[3] += _half_outer_size;
    }
}

inline void _thread_CX8(setStreamv2 *s, int *counter){
    for(ull i = 0; i < loop_size; i += _outer){
        inner_loop_func(_half_outer, s->rd, s->fd, s->fd_index, s->fd_off, counter);
        s->fd_off[0] += _half_outer_size;
        s->fd_off[1] += _half_outer_size;
        s->fd_off[2] += _half_outer_size;
        s->fd_off[3] += _half_outer_size;
        s->fd_off[4] += _half_outer_size;
        s->fd_off[5] += _half_outer_size;
        s->fd_off[6] += _half_outer_size;
        s->fd_off[7] += _half_outer_size;
    }
}

inline void set_up_lo(int ctrl, int targ){
    if(ctrl < targ){
        up_qubit = ctrl;
        lo_qubit = targ;
    }
    else{
        up_qubit = targ;
        lo_qubit = ctrl;
    }

    small_offset = 1ULL << (N-lo_qubit);
    half_small_offset = small_offset >> 1;
    large_offset = 1ULL << (N-up_qubit);
    half_large_offset = large_offset >> 1;
}

// fd_off[0] += size * sizeof(Type) afterward
void inner_loop(ull size, void *rd, int fd[1], int fd_index[1], ull fd_off[1], int *counter){

#ifdef MEMORY
    int batch_size = min(size / chunk_state, BATCH_SIZE);
#endif

    for (ull i = 0; i < size; i += chunk_state){

#ifdef MEMORY
        int now_idx = *counter % batch_size;
        if (now_idx == 0) {

            if (client[fd_index[0]].num_write) {
                poll_cq_batch(client + fd_index[0], client[fd_index[0]].num_write);
                client[fd_index[0]].num_write = 0;
            }

            for(int idx = 0; idx < batch_size; idx++) {
                if (fd_off[0] + idx * chunk_size < SNAPSHOT_SIZE) {
                    memcpy(client[fd_index[0]].buffer + idx * chunk_size, client[fd_index[0]].local_cache + fd_off[0] + idx * chunk_size, chunk_size);
                } else {
                    read_remote(client + fd_index[0], idx * chunk_size, fd_off[0] + idx * chunk_size, chunk_size, 0);
                    client[fd_index[0]].num_read++;
                }
            }

            if (client[fd_index[0]].num_read) {
                poll_cq_batch(client + fd_index[0], client[fd_index[0]].num_read);
                client[fd_index[0]].num_read = 0;
            }
        
        }
        if (gate_func((Type *)(client[fd_index[0]].buffer + now_idx * chunk_size))) {
            if (fd_off[0] < SNAPSHOT_SIZE) {
                memcpy(client[fd_index[0]].local_cache + fd_off[0], client[fd_index[0]].buffer + now_idx * chunk_size, chunk_size);
            } else {
                write_remote(client + fd_index[0], now_idx * chunk_size, fd_off[0], chunk_size, 0);
                client[fd_index[0]].num_write++;
            }
        }
#else
        if(pread (fd[0], rd, chunk_size, fd_off[0]));
        gate_func((Type *)rd);
        if(pwrite(fd[0], rd, chunk_size, fd_off[0]));
#endif

        fd_off[0] += chunk_size;
        *counter += 1;

        // printf("counter = %d\n", *counter);

    }
#ifdef MEMORY
//     if (client[fd_index[0]].num_write) {
//         poll_cq_batch(client + fd_index[0], client[fd_index[0]].num_write);
//         client[fd_index[0]].num_write = 0;
//     }
#endif
}

void inner_loop_read(ull size, void *rd, int fd[1], int fd_index[1], ull fd_off[1], int *counter){
    for (ull i = 0; i < size; i += chunk_state){

#ifdef MEMORY
        memcpy(rd, client[fd_index[0]].buffer + fd_off[0], chunk_size);
#else
        if(pread (fd[0], rd, chunk_size, fd_off[0]));
#endif

        gate_func((Type *)rd);
        fd_off[0] += chunk_size;
    }
}

// fd_off[0] += size * sizeof(Type) afterward
// fd_off[1] += size * sizeof(Type) afterward
void inner_loop2(ull size, void *rd, int fd[2], int fd_index[2], ull fd_off[2], int *counter){

    // printf("[Thread %d] size / chunk_state = %lld, file = %d %d, offset = %lld %lld\n", omp_get_thread_num(), size / chunk_state, fd_index[0], fd_index[1], fd_off[0], fd_off[1]);

#ifdef MEMORY
    int batch_size = min(size / chunk_state, BATCH_SIZE);
#endif

    for (ull i = 0; i < size; i += chunk_state){

#ifdef MEMORY
        int now_idx = *counter % batch_size;
        if (now_idx == 0) {
            for(int cnt = 0; cnt < 2; cnt++) {
                if (client[fd_index[cnt]].num_write) {
                    poll_cq_batch(client + fd_index[cnt], client[fd_index[cnt]].num_write);
                    client[fd_index[cnt]].num_write = 0;
                }
            }

            for(int cnt = 0; cnt < 2; cnt++)
                for(int idx = 0; idx < batch_size; idx++) {
                    if (fd_off[cnt] + idx * chunk_size < SNAPSHOT_SIZE) {
                        memcpy(client[fd_index[cnt]].buffer + (cnt * batch_size + idx) * chunk_size, client[fd_index[cnt]].local_cache + fd_off[cnt] + idx * chunk_size, chunk_size);
                    } else {
                        read_remote(client + fd_index[cnt], (cnt * batch_size + idx) * chunk_size, fd_off[cnt] + idx * chunk_size, chunk_size, 0);
                        client[fd_index[cnt]].num_read++;
                    }
                }

            for(int cnt = 0; cnt < 2; cnt++) {
                if (client[fd_index[cnt]].num_read) {
                    poll_cq_batch(client + fd_index[cnt], client[fd_index[cnt]].num_read);
                    client[fd_index[cnt]].num_read = 0;
                }
            }
        }
        memcpy(rd,            client[fd_index[0]].buffer + now_idx * chunk_size,                           chunk_size);
        memcpy(rd+chunk_size, client[fd_index[1]].buffer + now_idx * chunk_size + batch_size * chunk_size, chunk_size);
        if (gate_func((Type *)rd)){
            if (fd_off[0] < SNAPSHOT_SIZE) {
                memcpy(client[fd_index[0]].local_cache + fd_off[0], rd, chunk_size);
            } else {
                memcpy(client[fd_index[0]].buffer + now_idx * chunk_size, rd, chunk_size);
                write_remote(client + fd_index[0], now_idx * chunk_size, fd_off[0], chunk_size, 0);
                client[fd_index[0]].num_write++;
            }

            if (fd_off[1] < SNAPSHOT_SIZE) {
                memcpy(client[fd_index[1]].local_cache + fd_off[1], rd + chunk_size, chunk_size);
            } else {
                memcpy(client[fd_index[1]].buffer + now_idx * chunk_size + batch_size * chunk_size, rd + chunk_size, chunk_size);
                write_remote(client + fd_index[1], now_idx * chunk_size + batch_size * chunk_size, fd_off[1], chunk_size, 0);
                client[fd_index[1]].num_write++;
            }
        }
#else
        if(pread (fd[0], rd,            chunk_size, fd_off[0]));
        if(pread (fd[1], rd+chunk_size, chunk_size, fd_off[1]));
        gate_func((Type *)rd);
        if(pwrite(fd[0], rd,            chunk_size, fd_off[0]));
        if(pwrite(fd[1], rd+chunk_size, chunk_size, fd_off[1]));
#endif

        fd_off[0] += chunk_size;
        fd_off[1] += chunk_size;
        *counter += 1;
    }

#ifdef MEMORY
    // for(int cnt = 0; cnt < 2; cnt++) {
    //     if (client[fd_index[cnt]].num_write) {
    //         poll_cq_batch(client + fd_index[cnt], client[fd_index[cnt]].num_write);
    //         client[fd_index[cnt]].num_write = 0;
    //     }
    // }
#endif
}

void inner_loop2_read(ull size, void *rd, int fd[2], int fd_index[2], ull fd_off[2], int *counter){
    for (ull i = 0; i < size; i += chunk_state){

#ifdef MEMORY
        memcpy(rd,            client[fd_index[0]].buffer + fd_off[0], chunk_size);
        memcpy(rd+chunk_size, client[fd_index[1]].buffer + fd_off[1], chunk_size);
#else
        if(pread (fd[0], rd,            chunk_size, fd_off[0]));
        if(pread (fd[1], rd+chunk_size, chunk_size, fd_off[1]));
#endif

        gate_func((Type *)rd);
        fd_off[0] += chunk_size;
        fd_off[1] += chunk_size;
    }
}

void inner_loop2_swap(ull size, void *rd, int fd[2], int fd_index[2], ull fd_off[2], int *counter){
    
#ifdef MEMORY
    int batch_size = min(size / chunk_state, BATCH_SIZE);
#endif

    for (ull i = 0; i < size; i += chunk_state){

#ifdef MEMORY
        int now_idx = *counter % batch_size;
        if (now_idx == 0) {
            for(int cnt = 0; cnt < 2; cnt++) {
                if (client[fd_index[cnt]].num_write) {
                    poll_cq_batch(client + fd_index[cnt], client[fd_index[cnt]].num_write);
                    client[fd_index[cnt]].num_write = 0;
                }
            }

            for(int cnt = 0; cnt < 2; cnt++)
                for(int idx = 0; idx < batch_size; idx++) {
                    if (fd_off[cnt] + idx * chunk_size < SNAPSHOT_SIZE) {
                        memcpy(client[fd_index[cnt]].buffer + (cnt * batch_size + idx) * chunk_size, client[fd_index[cnt]].local_cache + fd_off[cnt] + idx * chunk_size, chunk_size);
                    } else {
                        read_remote(client + fd_index[cnt], (cnt * batch_size + idx) * chunk_size, fd_off[cnt] + idx * chunk_size, chunk_size, 0);
                        client[fd_index[cnt]].num_read++;
                    }
                }

            for(int cnt = 0; cnt < 2; cnt++) {
                if (client[fd_index[cnt]].num_read) {
                    poll_cq_batch(client + fd_index[cnt], client[fd_index[cnt]].num_read);
                    client[fd_index[cnt]].num_read = 0;
                }
            }
        }
        memcpy(rd,            client[fd_index[0]].buffer + now_idx * chunk_size,                           chunk_size);
        memcpy(rd+chunk_size, client[fd_index[1]].buffer + now_idx * chunk_size + batch_size * chunk_size, chunk_size);
        
        if (fd_off[0] < SNAPSHOT_SIZE) {
            memcpy(client[fd_index[0]].local_cache + fd_off[0], rd + chunk_size, chunk_size);
        } else {
            memcpy(client[fd_index[0]].buffer + now_idx * chunk_size,                           rd+chunk_size, chunk_size);
            write_remote(client + fd_index[0], now_idx * chunk_size, fd_off[0], chunk_size, 0);
            client[fd_index[0]].num_write++;
        }

        if (fd_off[1] < SNAPSHOT_SIZE) {
            memcpy(client[fd_index[1]].local_cache + fd_off[1], rd, chunk_size);
        } else {
            memcpy(client[fd_index[1]].buffer + now_idx * chunk_size + batch_size * chunk_size, rd,            chunk_size);
            write_remote(client + fd_index[1], now_idx * chunk_size + batch_size * chunk_size, fd_off[1], chunk_size, 0);
            client[fd_index[1]].num_write++;
        }
#else
        if(pread (fd[0], rd + chunk_size, chunk_size, fd_off[0]));
        if(pread (fd[1], rd,              chunk_size, fd_off[1]));
        if(pwrite(fd[0], rd,              chunk_size, fd_off[0]));
        if(pwrite(fd[1], rd + chunk_size, chunk_size, fd_off[1]));
#endif

        fd_off[0] += chunk_size;
        fd_off[1] += chunk_size;
        *counter += 1;
    }

#ifdef MEMORY
    // for(int cnt = 0; cnt < 2; cnt++) {
    //     if (client[fd_index[cnt]].num_write) {
    //         poll_cq_batch(client + fd_index[cnt], client[fd_index[cnt]].num_write);
    //         client[fd_index[cnt]].num_write = 0;
    //     }
    // }
#endif
}

void inner_loop4(ull size, void *rd, int fd[4], int fd_index[4], ull fd_off[4], int *counter){
    
#ifdef MEMORY
    int batch_size = min(size / chunk_state, BATCH_SIZE);
#endif

    for (ull i = 0; i < size; i += chunk_state){

#ifdef MEMORY
        int now_idx = *counter % batch_size;
        if (now_idx == 0) {
            for(int cnt = 0; cnt < 4; cnt++) {
                if (client[fd_index[cnt]].num_write) {
                    poll_cq_batch(client + fd_index[cnt], client[fd_index[cnt]].num_write);
                    client[fd_index[cnt]].num_write = 0;
                }
            }

            for(int cnt = 0; cnt < 4; cnt++)
                for(int idx = 0; idx < batch_size; idx++) {
                    if (fd_off[cnt] + idx * chunk_size < SNAPSHOT_SIZE) {
                        memcpy(client[fd_index[cnt]].buffer + (cnt * batch_size + idx) * chunk_size, client[fd_index[cnt]].local_cache + fd_off[cnt] + idx * chunk_size, chunk_size);
                    } else {
                        read_remote(client + fd_index[cnt], (cnt * batch_size + idx) * chunk_size, fd_off[cnt] + idx * chunk_size, chunk_size, 0);
                        client[fd_index[cnt]].num_read++;
                    }
                }

            for(int cnt = 0; cnt < 4; cnt++) {
                if (client[fd_index[cnt]].num_read) {
                    poll_cq_batch(client + fd_index[cnt], client[fd_index[cnt]].num_read);
                    client[fd_index[cnt]].num_read = 0;
                }
            }
        }
        for(int cnt = 0; cnt < 4; cnt++) {
            memcpy(rd + cnt * chunk_size, client[fd_index[cnt]].buffer + now_idx * chunk_size + cnt * batch_size * chunk_size, chunk_size);
        }
        if (gate_func((Type *)rd)){
            for(int cnt = 0; cnt < 4; cnt++) {
                if (fd_off[cnt] < SNAPSHOT_SIZE) {
                    memcpy(client[fd_index[cnt]].local_cache + fd_off[cnt], rd + cnt * chunk_size, chunk_size);
                } else {
                    memcpy(client[fd_index[cnt]].buffer + now_idx * chunk_size + cnt * batch_size * chunk_size, rd + cnt * chunk_size, chunk_size);
                    write_remote(client + fd_index[cnt], now_idx * chunk_size + cnt * batch_size * chunk_size, fd_off[cnt], chunk_size, 0);
                    client[fd_index[cnt]].num_write++;
                }
            }
        }
#else
        if(pread (fd[0], rd,                chunk_size, fd_off[0]));
        if(pread (fd[1], rd +   chunk_size, chunk_size, fd_off[1]));
        if(pread (fd[2], rd + 2*chunk_size, chunk_size, fd_off[2]));
        if(pread (fd[3], rd + 3*chunk_size, chunk_size, fd_off[3]));
        gate_func((Type *)rd);
        if(pwrite(fd[0], rd,               chunk_size, fd_off[0]));
        if(pwrite(fd[1], rd +   chunk_size, chunk_size, fd_off[1]));
        if(pwrite(fd[2], rd + 2*chunk_size, chunk_size, fd_off[2]));
        if(pwrite(fd[3], rd + 3*chunk_size, chunk_size, fd_off[3]));
#endif

        fd_off[0] += chunk_size;
        fd_off[1] += chunk_size;
        fd_off[2] += chunk_size;
        fd_off[3] += chunk_size;
        *counter += 1;
    }

#ifdef MEMORY
    // for(int cnt = 0; cnt < 4; cnt++) {
    //     if (client[fd_index[cnt]].num_write) {
    //         poll_cq_batch(client + fd_index[cnt], client[fd_index[cnt]].num_write);
    //         client[fd_index[cnt]].num_write = 0;
    //     }
    // }
#endif
}

void inner_loop8(ull size, void *rd, int fd[8], int fd_index[8], ull fd_off[8], int *counter){
    
#ifdef MEMORY
    int batch_size = min(size / chunk_state, BATCH_SIZE);
#endif

    void *rd0 = rd;
    void *rd1 = rd + 1*chunk_size;
    void *rd2 = rd + 2*chunk_size;
    void *rd3 = rd + 3*chunk_size;
    void *rd4 = rd + 4*chunk_size;
    void *rd5 = rd + 5*chunk_size;
    void *rd6 = rd + 6*chunk_size;
    void *rd7 = rd + 7*chunk_size;

    for (ull i = 0; i < size; i += chunk_state){

#ifdef MEMORY
        int now_idx = *counter % batch_size;
        if (now_idx == 0) {
            for(int cnt = 0; cnt < 8; cnt++) {
                if (client[fd_index[cnt]].num_write) {
                    poll_cq_batch(client + fd_index[cnt], client[fd_index[cnt]].num_write);
                    client[fd_index[cnt]].num_write = 0;
                }
            }

            for(int cnt = 0; cnt < 8; cnt++)
                for(int idx = 0; idx < batch_size; idx++) {
                    if (fd_off[cnt] + idx * chunk_size < SNAPSHOT_SIZE) {
                        memcpy(client[fd_index[cnt]].buffer + (cnt * batch_size + idx) * chunk_size, client[fd_index[cnt]].local_cache + fd_off[cnt] + idx * chunk_size, chunk_size);
                    } else {
                        read_remote(client + fd_index[cnt], (cnt * batch_size + idx) * chunk_size, fd_off[cnt] + idx * chunk_size, chunk_size, 0);
                        client[fd_index[cnt]].num_read++;
                    }
                }

            for(int cnt = 0; cnt < 8; cnt++) {
                if (client[fd_index[cnt]].num_read) {
                    poll_cq_batch(client + fd_index[cnt], client[fd_index[cnt]].num_read);
                    client[fd_index[cnt]].num_read = 0;
                }
            }
        }
        for(int cnt = 0; cnt < 8; cnt++) {
            memcpy(rd + cnt * chunk_size, client[fd_index[cnt]].buffer + now_idx * chunk_size + cnt * batch_size * chunk_size, chunk_size);
        }
        if (gate_func((Type *)rd)){
            for(int cnt = 0; cnt < 8; cnt++) {
                if (fd_off[cnt] < SNAPSHOT_SIZE) {
                    memcpy(client[fd_index[cnt]].local_cache + fd_off[cnt], rd + cnt * chunk_size, chunk_size);
                } else {
                    memcpy(client[fd_index[cnt]].buffer + now_idx * chunk_size + cnt * batch_size * chunk_size, rd + cnt * chunk_size, chunk_size);
                    write_remote(client + fd_index[cnt], now_idx * chunk_size + cnt * batch_size * chunk_size, fd_off[cnt], chunk_size, 0);
                    client[fd_index[cnt]].num_write++;
                }
            }
        }
#else
        if(pread (fd[0], rd0, chunk_size, fd_off[0]));
        if(pread (fd[1], rd1, chunk_size, fd_off[1]));
        if(pread (fd[2], rd2, chunk_size, fd_off[2]));
        if(pread (fd[3], rd3, chunk_size, fd_off[3]));
        if(pread (fd[4], rd4, chunk_size, fd_off[4]));
        if(pread (fd[5], rd5, chunk_size, fd_off[5]));
        if(pread (fd[6], rd6, chunk_size, fd_off[6]));
        if(pread (fd[7], rd7, chunk_size, fd_off[7]));
        gate_func((Type *)rd);
        if(pwrite(fd[0], rd0, chunk_size, fd_off[0]));
        if(pwrite(fd[1], rd1, chunk_size, fd_off[1]));
        if(pwrite(fd[2], rd2, chunk_size, fd_off[2]));
        if(pwrite(fd[3], rd3, chunk_size, fd_off[3]));
        if(pwrite(fd[4], rd4, chunk_size, fd_off[4]));
        if(pwrite(fd[5], rd5, chunk_size, fd_off[5]));
        if(pwrite(fd[6], rd6, chunk_size, fd_off[6]));
        if(pwrite(fd[7], rd7, chunk_size, fd_off[7]));
#endif

        fd_off[0] += chunk_size;
        fd_off[1] += chunk_size;
        fd_off[2] += chunk_size;
        fd_off[3] += chunk_size;
        fd_off[4] += chunk_size;
        fd_off[5] += chunk_size;
        fd_off[6] += chunk_size;
        fd_off[7] += chunk_size;
        *counter += 1;
    }

#ifdef MEMORY
    // for(int cnt = 0; cnt < 8; cnt++) {
    //     if (client[fd_index[cnt]].num_write) {
    //         poll_cq_batch(client + fd_index[cnt], client[fd_index[cnt]].num_write);
    //         client[fd_index[cnt]].num_write = 0;
    //     }
    // }
#endif
}

int isCtrl(int i , int ctrl, int n){
    int ctrl_mask = 1 << (n-ctrl-1);
    return i&ctrl_mask;
}

int iPair(const int i, const int targ, const int n){
    int targ_mask = 1 << (n-targ-1);
    return i^targ_mask;
}

void make_targ_pair(const int n, const int targ, int *pair_list){
    int idx = 0;
    int *checklist = (int *)malloc((1 << n) * sizeof(int));
    for (int i = 0; i < 1<<n; i += 1){
        checklist[i] = 0;
    }

    for (int i = 0; i < 1<<n; i += 1){
        if(checklist[i])
            continue;
        checklist[i] = 1;
        int p = iPair(i, targ, n);
        checklist[p] = 1;
        pair_list[idx++] = i;
        pair_list[idx++] = p;
    }

    // for (int i = 0; i < 1<<n; i += 1){
    //     printf("%d ", pair_list[i]);
    // }
    // printf("\n");
    // printf("in make_targ_pair\n");
    free(checklist);
}

void make_work_pair(const int n, const int ctrl, const int targ, int *pair_list){
    int idx = 0;
    int *checklist = (int *)malloc((1 << n) * sizeof(int));
    for (int i = 0; i < 1<<n; i += 1){
        checklist[i] = 0;
    }

    for (int i = 0; i < 1<<n; i += 1){
        if(checklist[i])
            continue;
        checklist[i] = 1;
        if (ctrl < 0 && targ >= 0){
            int p = iPair(i, targ, n);
            checklist[p] = 1;
            pair_list[idx++] = i < p ? i : p;
            pair_list[idx++] = i < p ? p : i;
        }
        else if (isCtrl(i, ctrl, n)){
            if(targ < 0){
                pair_list[idx++] = i;
                pair_list[idx++] = i;
            }
            else{
                int p = iPair(i, targ, n);
                checklist[p] = 1;
                pair_list[idx++] = i < p ? i : p;
                pair_list[idx++] = i < p ? p : i;
            }
        }
    }
    // for (int i = 0; i < 1<<n; i += 1){
    //     printf("%d ", pair_list[i]);
    // }
    // printf("\n");

    free(checklist);
}

void make_work_group(const int n, const int q0, const int q1, int *group_list){
    int idx = 0;
    int *checklist = (int *)malloc((1 << n) * sizeof(int));
    for (int i = 0; i < 1<<n; i += 1){
        checklist[i] = 0;
    }

    for (int i = 0; i < 1<<n; i += 1){
        if(checklist[i])
            continue;
        int m00 = i;
        int m01 = iPair(i, q1, n);
        int m10 = iPair(i, q0, n);
        int m11 = iPair(m10, q1, n);
        checklist[m00] = 1;
        checklist[m01] = 1;
        checklist[m10] = 1;
        checklist[m11] = 1;
        group_list[idx++] = m00;
        group_list[idx++] = m01;
        group_list[idx++] = m10;
        group_list[idx++] = m11;
    }

    // for (int i = 0; i < 1<<n; i += 1){
    //     printf("%d ", group_list[i]);
    // }
    // printf("\n");
    // fflush(stdout);

    free(checklist);
}

void make_work_warp(const int n, const int q0, const int q1, const int q2, int *warp_list){
    int idx = 0;
    int *checklist = (int *)malloc((1 << n) * sizeof(int));
    for (int i = 0; i < 1<<n; i += 1){
        checklist[i] = 0;
    }

    for (int i = 0; i < 1<<n; i += 1){
        if(checklist[i])
            continue;
        int m000 = i;
        int m001 = iPair(i, q2, n);
        int m010 = iPair(i, q1, n);
        int m011 = iPair(m010, q2, n);
        int m100 = iPair(i, q0, n);
        int m101 = iPair(m100, q2, n);
        int m110 = iPair(m100, q1, n);
        int m111 = iPair(m110, q2, n);

        checklist[m000] = 1;
        checklist[m001] = 1;
        checklist[m010] = 1;
        checklist[m011] = 1;
        checklist[m100] = 1;
        checklist[m101] = 1;
        checklist[m110] = 1;
        checklist[m111] = 1;
        warp_list[idx++] = m000;
        warp_list[idx++] = m001;
        warp_list[idx++] = m010;
        warp_list[idx++] = m011;
        warp_list[idx++] = m100;
        warp_list[idx++] = m101;
        warp_list[idx++] = m110;
        warp_list[idx++] = m111;

    }

    // for (int i = 0; i < 1<<n; i += 1){
    //     printf("%d ", warp_list[i]);
    // }
    // printf("\n");
    // fflush(stdout);

    free(checklist);
}
