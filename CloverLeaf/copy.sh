#!/bin/bash
for i in `seq 1 $1`;
do
	scp  trace.sh ubuntu@node$i:/home/ubuntu/CloverLeaf_CUDA-master
done 
