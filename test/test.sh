#!/bin/bash

rm -f *.out
rm -f *.log
rm -f *.json
rm -f compare*

cat << EOF1 > tt.json
{
   "ref_file": "../samples/tt.ref",
   "verify": {
      "against": "whizard",
      "epsilon": 0.0001
   },
   "check_cpu": true,
   "warmup": {
      "n_trials": 10,
      "n_events": 10000
   },
   "gpu_memory": 5000, 
   "msq": {
      "threads": 512
   },
   "create_boosts": {
      "threads": 512
   },
   "apply_boosts": {
      "threads": 512
   }
}
EOF1

cat << EOF2 > ttH.json
{
   "ref_file": "../samples/ttH.ref",
   "verify": {
      "against": "whizard",
      "epsilon": 0.0001
   },
   "check_cpu": true,
   "warmup": {
      "n_trials": 10,
      "n_events": 10000
   },
   "gpu_memory": 5000, 
   "msq": {
      "threads": 512
   },
   "create_boosts": {
      "threads": 512
   },
   "apply_boosts": {
      "threads": 512
   }
}
EOF2

cat << EOF3 > uu4g.json
{
   "ref_file": "../samples/uu4g.ref",
   "verify": {
      "against": "whizard",
      "epsilon": 0.0001
   },
   "check_cpu": true,
   "warmup": {
      "n_trials": 10,
      "n_events": 10000
   },
   "gpu_memory": 5000, 
   "msq": {
      "threads": 512
   },
   "create_boosts": {
      "threads": 512
   },
   "apply_boosts": {
      "threads": 512
   }
}
EOF3

testcases="tt ttH uu4g"

echo "CHECK AGAINST WHIZARD REFERENCE MOMENTA"
for t in $testcases; do
   d=$(date +%y_%m_%_H_%M_%s)
   ../phs.x $t.json | tee ${t}_${d}.out
   mv cuda.log cuda_${t}_${d}.log
   mv input.log input_${t}_${d}.log
   if [ -f compare.gpu ]; then
     mv compare.gpu compare_${t}_${d}.gpu
   fi
   if [ -f compare.cp ]; then
     mv compare.cpu compare_${t}_${d}.cpu
   fi
done

echo "CHECK INTERNALL AGAINST CPU IMPLEMENTATION"
for t in $testcases; do
   sed -i 's/whizard/internal/g' $t.json
   ../phs.x $t.json
done
