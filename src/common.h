#ifndef COMMON_H_
#define COMMON_H_
#include <sys/time.h>
#include <mpi.h>

// typedef
typedef unsigned long long ull;
typedef double Type_t;
typedef struct {
    Type_t real;
    Type_t imag;
} Type;

// global variable, Remark: 第0位Qubit是最高位
extern int N; // total_qubit
extern int mpi_segment;
extern int thread_segment;
extern int file_segment;
extern int para_segment;
extern int chunk_segment;
extern unsigned int max_path;
//extern unsigned int MAX_QUBIT; // FP16
extern unsigned int max_depth; // 最多一千的 gates
extern char **state_paths;
extern int *fd_arr; //int fd_arr [num_file];
extern int **fd_arr_set; //int fd_arr_set [set][num_file];
extern Type *q_read;
// extern Type *q_write;
extern int rank;
extern int world_size;
extern int IsDensity;
extern int IsDirectIO;
extern int SkipInithread_state;
extern int SetOfSaveState;

extern ull num_file; // for extending to more devices
extern ull num_thread; // number of thread
extern ull half_num_thread;
extern ull quarter_num_thread;
extern ull eighth_num_thread;
extern ull num_thread_per_file;  // applying threads to files
extern ull file_state; // number of states in single file
extern ull thread_state; // number of states in single thread
extern ull file_size;
extern ull thread_size;
extern ull chunk_state; // unit: Type
extern ull chunk_size;
extern ull buffer_size; // buffer_size for each thread

// macro
// usually use
#define PI 3.14159265358979

// time measure
#define MEASURET_START \
    struct timeval start; \
    struct timeval end; \
    unsigned long diff; \
    gettimeofday(&start,NULL);

#define MEASURET_END \
    gettimeofday(&end,NULL); \
    diff = 1000000 * (end.tv_sec-start.tv_sec) + \
    end.tv_usec-start.tv_usec; \
    MPI_Barrier(MPI_COMM_WORLD); \
    printf("Rank: %d %ld (us)\n", rank,diff);

#define isFile(target) ((target>=0)&&(target < file_segment) ? 1 : 0)
#define isThread(target) ((target >= file_segment) && (target < thread_segment) ? 1 : 0)
#define isMiddle(target) ((target >= thread_segment) && (target < N-chunk_segment) ? 1 : 0)
#define isLocal(target) ((target >= N - chunk_segment) ? 1 : 0)
#define isMpi(target) ((target < mpi_segment) ? 1 : 0)
int file_exists(char *filename);
int mk_dir(char *dir);
void run_simulator();

#endif
