#!/bin/bash

if [ "$#" -lt 3 ]; then
    echo "Usage: numNodes firstImg lastImg"
    exit
fi


imgsPerNode=$(expr $(expr $(expr $3 - $2) + 1) / $1)
last=$2
maxNode=$(expr $1 - 1)

SECONDS=0
for i in `seq 0 $maxNode`;
do
  first=$last
  last=$(expr $first + $imgsPerNode)
  command="./run_alexnet $first $last > output$i.txt 2>&1"
  ssh ubuntu@node$i $command &
done
wait
duration=$SECONDS

output=""
for i in `seq 0 $maxNode`;
do
  echo -e "NODE $i OUTPUT \n -----------------------" >> ~/TX1-storage/results/result.txt
  ssh ubuntu@node$i "cat /home/ubuntu/output$i.txt >> ~/TX1-storage/results/result.txt"
  echo -e "\n\n\n" >> ~/TX1-storage/results/result.txt
done


echo $duration  >> ~/TX1-storage/results/result.txt
echo $duration
exit
