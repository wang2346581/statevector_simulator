#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <fcntl.h>
#include <unistd.h>
#include <omp.h>
#include "common.h"
#include "gate.h"
#include "gate_util.h"
#include "gate_chunk.h"
#include "measure.h"
#include "mpi.h"

int* measure_fd_arr;

void measure(int targ, int fd_set){
    int t = omp_get_thread_num();
    setStreamv2 *s = &thread_settings[t];
    s->id=t;
    ull targShift = mpi_segment-targ-1;
    ull targSegment = 1 << targShift;

    /*----------------------------
    [1st phase] Setting up global variables.
    ----------------------------*/
    if (t == 0){
        measure_fd_arr = fd_arr_set[fd_set];
        gate_func = PreMeasure;

        loop_size = thread_state;
        if(isMpi(targ))
        {
            gate_func = PreMeasureMPI;
            inner_loop_func = inner_loop_read;
            gate_size = chunk_state;
            _outer = thread_state;
        }
        else
        {
            targ=targ-mpi_segment;
            if (isLocal(targ)){
            inner_loop_func = inner_loop_read;
            gate_move.half_targ = (1ULL << (N-targ-1));
            gate_size = chunk_state;
            }
            else{
                inner_loop_func = inner_loop2_read;
                gate_move.half_targ = chunk_state;
                gate_size = 2 * chunk_state;
            }
            if(isMiddle(targ)){
                targ_offset = 1 << (N-targ);
                half_targ_offset = targ_offset >> 1;
                set_outer(targ_offset);
            }
            else{
                _outer = thread_state;
            }

            if(isFile(targ)){
                make_targ_pair(file_segment, targ, fd_pair);
            }
            if(isThread(targ)){
                make_targ_pair(para_segment, targ-file_segment, td_pair);
            }

            // printf("Thread rank: %d pass barrier, targ= %d\n", t, targ);
            targ=targ+mpi_segment;
        }
        
        real[0] = 0.0;
        real[1] = 0.0;
    }

    #pragma omp barrier

    /*------------------------------
    [2nd phase] Applying measure
    ------------------------------*/
    if ((isFile(targ-mpi_segment) || isThread(targ-mpi_segment)) && t >= half_num_thread){
        #pragma omp barrier
        #pragma omp barrier
        return;
    }
    if(isMpi(targ))
    {
        int f = t/num_thread_per_file;
        s->fd[0] = measure_fd_arr[f];
        s->fd_off[0] = 0;
        _thread_CX(s);
    }
    else if (isFile(targ-mpi_segment)){
        int f = t/num_thread_per_file;
        int td = t%num_thread_per_file;

        int fd0 = fd_pair[2*f];
        int fd1 = fd_pair[2*f+1];

        ull t_off = td * thread_state * sizeof(Type);

        s->fd[0] = measure_fd_arr[fd0]; s->fd_off[0] = t_off;
        s->fd[1] = measure_fd_arr[fd1]; s->fd_off[1] = t_off;
        _thread_CX(s);
    }

    else if (isThread(targ-mpi_segment)) {
        int fd = t/(num_thread_per_file/2);
        int td = t%(num_thread_per_file/2);

        int t0 = td_pair[2*td];
        int t1 = td_pair[2*td + 1];

        ull t0_off = t0 * thread_state * sizeof(Type);
        ull t1_off = t1 * thread_state * sizeof(Type);

        s->fd[0] = measure_fd_arr[fd]; s->fd_off[0] = t0_off;
        s->fd[1] = measure_fd_arr[fd]; s->fd_off[1] = t1_off;
        _thread_CX(s);
    }

    else if(isMiddle(targ-mpi_segment)){
        int fd = t/num_thread_per_file;
        int td = t%num_thread_per_file;

        ull t0_off = td * thread_state * sizeof(Type);
        ull t1_off = t0_off + half_targ_offset * sizeof(Type);

        s->fd[0] = measure_fd_arr[fd]; s->fd_off[0] = t0_off;
        s->fd[1] = measure_fd_arr[fd]; s->fd_off[1] = t1_off;
        _thread_CX2(s);
    }

    else if (isLocal(targ-mpi_segment)){
        int fd = t/num_thread_per_file;
        int td = t%num_thread_per_file;

        ull t_off = td * thread_state * sizeof(Type);

        s->fd[0] = measure_fd_arr[fd]; s->fd_off[0] = t_off;
        _thread_CX(s);
    }
    #pragma omp barrier

    /*----------------------------
    [3rd phase] Setting up global variables.
    ----------------------------*/
    // MPI_Barrier(MPI_COMM_WORLD);

    if (t == 0){
        // printf("target %d = 0, with probability %lf\n", targ, real[0]);
        // printf("target %d = 1, with probability %lf\n\n", targ, real[1]);
        // fflush(stdout);
        if(world_size!=1)
        {
            if(isMpi(targ))
            {
                if((rank & targSegment)==targSegment)
                {
                    real[1]=real[0];
                    real[0]=0.0;
                }
            }
            // printf("rank%d, real[0]: %lf\n", rank, real[0]);
            // printf("rank%d, real[1]: %lf\n", rank, real[1]);
            double sum_real0 = 0.0;
            double sum_real1 = 0.0;
            MPI_Allreduce(&real[0], &sum_real0, 1, MPI_DOUBLE, MPI_SUM,MPI_COMM_WORLD);
            MPI_Allreduce(&real[1], &sum_real1, 1, MPI_DOUBLE, MPI_SUM,MPI_COMM_WORLD);
            
            // if(rank == 0) {
            //     MPI_Bcast(&sum_real0, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            //     MPI_Bcast(&sum_real1, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            // } else {
            //     MPI_Bcast(&sum_real0, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            //     MPI_Bcast(&sum_real1, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            // }
            real[0]=sum_real0;
            real[1]=sum_real1;
            // if(rank==0)
            // {
            //     for(int i=1;i<world_size;i++) //Recv everyone's real[0] and real[1]
            //     {
            //         double tmp_real0=0;
            //         double tmp_real1=0;
            //         MPI_Recv(&tmp_real0,1,MPI_DOUBLE,i,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
            //         MPI_Recv(&tmp_real1,1,MPI_DOUBLE,i,1,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
            //         real[0]+=tmp_real0;
            //         real[1]+=tmp_real1;
            //     }
            //     for(int i=1;i<world_size;i++)//Send everyone 
            //     {
            //         MPI_Send(&real[0],1,MPI_DOUBLE,i,0,MPI_COMM_WORLD);
            //         MPI_Send(&real[1],1,MPI_DOUBLE,i,1,MPI_COMM_WORLD);
            //     }
            // }
            // else //Send real[0] and real[1] and get global real[0] and real[1]
            // {
            //     MPI_Send(&real[0],1,MPI_DOUBLE,0,0,MPI_COMM_WORLD);
            //     MPI_Send(&real[1],1,MPI_DOUBLE,0,1,MPI_COMM_WORLD);   
            //     MPI_Recv(&real[0],1,MPI_DOUBLE,0,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
            //     MPI_Recv(&real[1],1,MPI_DOUBLE,0,1,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
            // }
        }
        // if(!rank)
        // {
        //     printf("real[0] %lf\n",real[0]);
        //     printf("real[1] %lf\n",real[1]);
        // }
        // if(!rank){
        //     printf("real[0]: %lf\n", real[0]);
        //     printf("real[1]: %lf\n\n", real[1]);
        // }
        double d;
        d = (double) rand() / ((double) RAND_MAX + 1);
        d *= (real[0]+real[1]);
        MPI_Bcast(&d, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        // if(rank==0)
        // {
        //     for(int i=1;i<world_size;i++)
        //     {
        //         MPI_Send(&d,1,MPI_DOUBLE,i,2,MPI_COMM_WORLD);
        //     }
        // }
        // else
        // {
        //     MPI_Recv(&d,1,MPI_DOUBLE,0,2,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        // }

        // if(real[0] > 0.99999){
        //     printf("[1]");
        //     // printf("[0]target %d = 0", targ);
        //     // fflush(stdout);
        //     gate_func = Measure_0;
        //     real[0] = 1;
        //     real[1] = 0;
        // }
        // else if(real[1] > 0.99999){
        //     printf("[2]");
        //     // printf("[1]target %d = 1", targ);
        //     // fflush(stdout);
        //     gate_func = Measure_1;
        //     real[0] = 1;
        //     real[1] = 1;
        // }
        if(d < real[0]){
            // printf("[3]");
            // printf("[0]target %d = 0", targ);
            // fflush(stdout);
            gate_func = Measure_0;
            real[0] = 1/sqrt(real[0]);
            real[1] = 0;
            if(isMpi(targ) &&((rank & targSegment)==targSegment))
            {
                real[0]=0;
            }
        }
        else{
            // printf("[4]");
            // printf("[1]target %d = 1", targ);
            // fflush(stdout);
            gate_func = Measure_1;
            real[0] = 1/sqrt(real[1]);
            real[1] = 1;
            if(isMpi(targ) && ((rank & targSegment)==0))
            {
                real[0]=0;
            }
        }
        if (isLocal(targ-mpi_segment)){
            inner_loop_func = inner_loop;
        }
        else if(isMpi(targ)){
            inner_loop_func = inner_loop;
        }
        else {
            inner_loop_func = inner_loop2;
        }
    }
    #pragma omp barrier

    /*------------------------------
    [4th phase] Applying measure
    ------------------------------*/
    if(isMpi(targ))
    {
        int f = t/num_thread_per_file;
        s->fd[0] = measure_fd_arr[f];
        s->fd_off[0] = 0;
        Type *rd=(Type *)s->rd;
        // _thread_CX(s);
        for (ull i = 0; i < thread_state; i += chunk_state)
        {
            if(pread (s->fd[0], s->rd, chunk_size, s->fd_off[0]));
            for(int i=0;i<gate_size;i++)
            {
                rd[i].real*=real[0];
                rd[i].imag*=real[0];
            }
            if(pwrite(s->fd[0], s->rd, chunk_size, s->fd_off[0]));
            s->fd_off[0] += chunk_size;
        }
        return;
    }
    if (isFile(targ-mpi_segment)){
        int f = t/num_thread_per_file;
        int td = t%num_thread_per_file;

        int fd1 = fd_pair[2*f];
        int fd2 = fd_pair[2*f+1];

        ull t_off = td * thread_state * sizeof(Type);

        s->fd[0] = measure_fd_arr[fd1]; s->fd_off[0] = t_off;
        s->fd[1] = measure_fd_arr[fd2]; s->fd_off[1] = t_off;
        _thread_CX(s);
        return;
    }

    if (isThread(targ-mpi_segment)) {
        int fd = t/(num_thread_per_file/2);
        int td = t%(num_thread_per_file/2);

        int t0 = td_pair[2*td];
        int t1 = td_pair[2*td + 1];

        ull t0_off = t0 * thread_state * sizeof(Type);
        ull t1_off = t1 * thread_state * sizeof(Type);

        s->fd[0] = measure_fd_arr[fd]; s->fd_off[0] = t0_off;
        s->fd[1] = measure_fd_arr[fd]; s->fd_off[1] = t1_off;
        _thread_CX(s);
        return;
    }

    if(isMiddle(targ-mpi_segment)){
        int fd = t/num_thread_per_file;
        int td = t%num_thread_per_file;

        ull t0_off = td * thread_state * sizeof(Type);
        ull t1_off = t1_off + half_targ_offset * sizeof(Type);

        s->fd[0] = measure_fd_arr[fd]; s->fd_off[0] = t0_off;
        s->fd[1] = measure_fd_arr[fd]; s->fd_off[1] = t1_off;
        _thread_CX2(s);
        return;
    }

    if (isLocal(targ-mpi_segment)){
        int fd = t/num_thread_per_file;
        int td = t%num_thread_per_file;

        ull t_off = td * thread_state * sizeof(Type);

        s->fd[0] = measure_fd_arr[fd]; s->fd_off[0] = t_off;
        _thread_CX(s);
        return;
    }

    // shouldn't be here, previous states has been saved.
    printf("error gate instruction: should not be here.\n");
    exit(0);
}

void save_state(int fd_set_src, int fd_set_dst){
    int t = omp_get_thread_num();
    int fd = t/num_thread_per_file;
    int td = t%num_thread_per_file;

    int fd_src = fd_arr_set[fd_set_src][fd];
    int fd_dst = fd_arr_set[fd_set_dst][fd];

    ull t_off = td * thread_state * sizeof(Type);

    // void* rd = (void *) q_read + t * chunk_size;
    void* rd = thread_settings[t].rd;
    for(ull i = 0; i < thread_state; i += chunk_state){
        if(pread  (fd_src, rd, chunk_size, t_off));
        if(pwrite (fd_dst, rd, chunk_size, t_off));
        t_off += chunk_size;
    }
    return;
}