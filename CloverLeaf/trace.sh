#!/bin/bash

source /home/ubuntu/extrae-install/etc/extrae.sh

export EXTRAE_CONFIG_FILE=./extrae.xml
export LD_PRELOAD=${EXTRAE_HOME}/lib/libcudampitrace.so
#export LD_PRELOAD=${EXTRAE_HOME}/lib/libcudampitracef.so

## Run the desired program
#$*
#
/home/ubuntu/CloverLeaf_CUDA-master/clover_leaf
