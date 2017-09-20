#!/bin/bash


echo reza 
echo reza
echo reza
source /home/ubuntu/extrae-install/etc/extrae.sh
let i=$1-1

echo $i 
echo $i
./copy-rm.sh $i

${EXTRAE_HOME}/bin/mpi2prv -f TRACE.mpits -o cloverleaf-$1node.prv

mkdir temp
mv TRACE* ./temp
mv set-0 ./temp
mv cloverleaf-* ./temp
mv temp $1node

