#!/bin/bash

set -e
# set -v
tryTime=2
shPath=`pwd`
fileDir="./build/bin"

judgeFileNameSuffix=".*flags"
isBaseline0=1
isBaseline1=1
# baseline0=0
# baseline1=0

rm -rf time0.log
rm -rf time1.log
rm -rf stepAcc0.log
rm -rf stepAcc1.log
rm -rf totalAcc0.log
rm -rf totalAcc1.log

for eachfile in "$fileDir"/* 
do
   echo "Filename: $eachfile"
   tmpSum=0
   for ((j = 0 ; j < $tryTime ; j++)); do
        tmp=`$eachfile | awk '{if($1 ~ /^[0-9]+$/) print $1;}'`
        let "tmpSum += tmp"
        echo "  tryTime: $j: $tmp $tmpSum"
   done
   # let "avg = tmpSum/tryTime" # bash only int
   avg=`echo "scale=2; $tmpSum/$tryTime" | bc`
   echo "   avgTime: $avg"
   # https://stackoverflow.com/questions/18709962/regex-matching-in-a-bash-if-statement
   if [[ $eachfile =~ $judgeFileNameSuffix ]]; then
      echo "   match Suffix!"
      if [[ $isBaseline0 == 1 ]]; then
         echo "   write baseline 0"
         baseline0=$avg
         beforeTime0=$avg
         isBaseline0=0       
      fi
      echo "${avg}" >> time0.log
      stepAcc=`echo "scale=2; $beforeTime0/$avg" | bc`
      beforeTime1=$avg
      totalAcc=`echo "scale=2; $baseline0/$avg" | bc`
      echo "   stepAcc: $stepAcc, totalAcc: $totalAcc"
      echo "$stepAcc" >> stepAcc0.log
      echo "$totalAcc" >> totalAcc0.log
   else
      echo "   Not match Suffix!"
      if [[ $isBaseline1 == 1 ]]; then
         echo "   write baseline 1"
         baseline1=$avg
         beforeTime1=$avg
         isBaseline1=0
      fi
      echo "${avg}" >> time1.log
      stepAcc=`echo "scale=2; $beforeTime1/$avg" | bc`
      beforeTime1=$avg
      totalAcc=`echo "scale=2; $baseline1/$avg" | bc`
      echo "   stepAcc: $stepAcc, totalAcc: $totalAcc"
      echo "$stepAcc" >> stepAcc1.log
      echo "$totalAcc" >> totalAcc1.log
   fi
done