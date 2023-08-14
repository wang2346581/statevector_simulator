// scheduler.h

#ifndef SCHED_H
#define SCHED_H
// void check_gateMap (int total_gate);
// void check_qubitTime (int total_qubit, int total_gate);
void circuit_scheduler (gate *gateMap, int (*qubitTime)[MAX_DEPTH], int total_qubit, int total_gate);

#endif