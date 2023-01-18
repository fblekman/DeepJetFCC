#!/bin/bash

#source_dir='/afs/cern.ch/work/k/kgautam/public/ntuplesDeepJet/p8_ee_Zuds_ecm91_3M/'

processes=("p8_ee_Zuds_ecm91" "p8_ee_Zcc_ecm91" "p8_ee_Zbb_ecm91")

base_dir="/afs/cern.ch/user/e/eploerer/private/DeepJetFCC/DeepJetFCC/"

source_dir="/eos/user/e/eploerer/DeepJet_sourceFiles/short_files_12012023/"

cd ${source_dir}${processes[2]}
ls *.root > ${processes[2]}.txt
file=${source_dir}${processes[2]}/${processes[2]}.txt
cd ..
if [ ! -d "tmp" ]
then
mkdir "tmp"
fi
i=0
while read line; do
cd ${source_dir}
echo Starting file ${i} at `date`...
hadd -f tmp/chunk_${i}.root ${processes[0]}/chunk_${i}.root ${processes[0]}/chunk_$((i+10)).root ${processes[0]}/chunk_$((i+20)).root ${processes[1]}/chunk_${i}.root ${processes[2]}/chunk_${i}.root
#echo tmp/chunk_${i}.root ${processes[0]}/chunk_${i}.root ${processes[0]}/chunk_$((i+10)).root ${processes[0]}/chunk_$((i+20)).root ${processes[1]}/chunk_${i}.root ${processes[2]}/chunk_${i}.root
cd ${base_dir}
./postprocessor_2d.exe ${source_dir}tmp/chunk_${i}.root ${source_dir}tmp/
i=$((i+1))
done < $file



