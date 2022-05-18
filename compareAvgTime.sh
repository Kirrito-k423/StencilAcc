#!/bin/bash

set -e
# set -v
tryTime=3
shPath=`pwd`
fileDir="./build/bin"

for eachfile in "$fileDir"/* 
do
   echo "Filename: $eachfile"
   tmpSum=0
   for ((j = 0 ; j < $tryTime ; j++)); do
        tmp=`$eachfile | awk '{if($1 ~ /^[0-9]+$/) print $1;}'`
        let "tmpSum += tmp"
        echo "  tryTime: $j: $tmp $tmpSum"
   done
   let "avg = tmpSum/tryTime"
   echo "   avgTime: $avg"
done