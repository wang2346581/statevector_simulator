#!/bin/bash
NUM_THREAD=16
for i in $(seq 0 $((${NUM_THREAD} - 1)))
do
    ./rdma_server $((20886 + ${i})) &
done
