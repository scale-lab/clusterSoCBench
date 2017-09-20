#!/bin/bash
mkdir profile
mv profile.*.*.* ./profile
for i in `seq 1 $1`;
do
	scp ubuntu@node$i:/home/ubuntu/CloverLeaf_CUDA-master/profile.*.*.* ./profile
done 
