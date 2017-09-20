#!/bin/bash
for i in `seq 1 $1`;
do
	scp ubuntu@node$i:/home/ubuntu/CloverLeaf_CUDA-master/set-0/* /home/ubuntu/CloverLeaf_CUDA-master/set-0/
        ssh node$i 'rm /home/ubuntu/CloverLeaf_CUDA-master/set-0/*'
done 
