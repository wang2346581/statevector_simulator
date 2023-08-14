#ifndef GATE_H_
#define GATE_H_
#include <mpi.h>
// typedef
typedef enum gate_ops{
    GATE_H = 0,
    GATE_S = 1,
    GATE_T = 2,
    GATE_X = 3,
    GATE_Y = 4,
    GATE_Z = 5,
    GATE_Phase = 6,
    GATE_U1 = 7,
    GATE_CX = 8,
    GATE_CY = 9,
    GATE_CZ = 10,
    GATE_CPhase = 11,
    GATE_CU1 = 12,
    GATE_SWAP = 13,
    GATE_TOFFOLI = 14,
    OPS_MEASURE = 20,
    OPS_MEASURE_MULTI = 21,
    OPS_COPY = 22,
    GATE_U2 = 31,
    GATE_U3 = 32
} GATE_OPS;

typedef struct gate {
    // action should only be used in scheduler.
    // for normal use it is unnecessary
    int active; // wheter this gate is combine

    GATE_OPS gate_ops; // gate's opcode
    int numCtrls; // #control
    int numTargs; // #targets
    int val_num; // #variable does the gate has
    int ctrls [3]; // at most three
    int targs [3]; // at most three
    Type_t *real_matrix; // angle (real) also put in here 可能(?)
    Type_t *imag_matrix; // row-maj
} gate;

typedef struct setStreamv2 {
    int id;             // thread id
    unsigned int partner_rank[8];      // correspond rank for mpi
    int fd[8];          // 要處理的file對應的file descriptor
    unsigned long long fd_off[8];      // 要處理的file內的offset
    void *rd;           // 指向buffer的位置
    MPI_Request request[15];
} setStreamv2;
extern setStreamv2 *thread_settings;

// global variable
extern unsigned int total_gate;
extern gate *gateMap; // gate gateMap [MAX_QUBIT*max_depth];
// extern int **qubitTime; // int qubitTime [MAX_QUBIT][max_depth];

extern Type_t *real;
extern Type_t *imag;

void single_gate(int targ, int ops, int density);
void control_gate(int ctrl, int targ, int ops, int density);
void unitary4x4(int q0, int q1, int density);
void SWAP(int q0, int q1, int density);
void unitary8x8(int q0, int q1, int q2, int density);
void print_gate(gate* g);

#endif
