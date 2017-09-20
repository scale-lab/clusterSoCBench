#!/bin/bash
for i in `seq 1 $1`;
do
	ssh ubuntu@node$i 'rm /home/ubuntu/CloverLeaf_CUDA-master/profile.*.*.*' 
done 
