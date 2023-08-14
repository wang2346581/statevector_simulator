#ifndef GATE_CHUNK_H_
#define GATE_CHUNK_H_

#include <stdbool.h>

bool H_gate (Type *q_rd);
bool S_gate (Type *q_rd); bool Sc_gate (Type *q_rd);
bool T_gate (Type *q_rd); bool Tc_gate (Type *q_rd);
bool X_gate (Type *q_rd);
bool Y_gate (Type *q_rd); bool Yc_gate (Type *q_rd);
bool Z_gate (Type *q_rd);
bool P_gate (Type *q_rd); bool Pc_gate (Type *q_rd);
bool U_gate (Type *q_rd); bool Uc_gate (Type *q_rd);

bool X_gate2 (Type *q_rd);
bool Y_gate2 (Type *q_rd); bool Yc_gate2 (Type *q_rd);
bool Z_gate2 (Type *q_rd);
bool P_gate2 (Type *q_rd); bool Pc_gate2 (Type *q_rd);

bool U_gate2 (Type *q_rd); bool Uc_gate2 (Type *q_rd);
bool U2_gate (Type *q_rd); bool U2c_gate (Type *q_rd);
bool SWAP_gate (Type *q_rd);
bool U3_gate (Type *q_rd); bool U3c_gate (Type *q_rd);

bool PreMeasure (Type *q_rd);
bool Measure_0 (Type *q_rd);
bool Measure_1 (Type *q_rd);

extern bool (*gate_ops[16][2])(Type *);

#endif
