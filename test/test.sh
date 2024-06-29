#!/bin/bash

rm -f *.out
rm -f *.log
rm -f *.json
rm -f compare*

generate_json () {
proc=$1
cat << EOF > ${proc}.json
{
   "ref_file": "../samples/${proc}.ref",
   "verify": {
      "against": "whizard",
      "epsilon": 0.0001
   },
   "check_cpu": true,
   "warmup": {
      "n_trials": 10,
      "n_events": 10000
   },
   "gpu_memory": 11000, 
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
EOF

}


testcases="tt ttH uu4g bwbw bmunubmunu"

echo "CHECK AGAINST WHIZARD REFERENCE MOMENTA"
for t in $testcases; do
   generate_json $t
   echo "TESTCASE: $t"
   d=$(date +%m_%d_%_H_%M_%s)
   ../phs.x $t.json | tee ${t}_${d}.out
   mv cuda.log cuda_${t}_${d}.log
   mv input.log input_${t}_${d}.log
   if [ -f compare.gpu ]; then
     mv compare.gpu compare_${t}_${d}.gpu
   fi
   if [ -f compare.cpu ]; then
     mv compare.cpu compare_${t}_${d}.cpu
   fi
done

echo "CHECK INTERNALLY AGAINST CPU IMPLEMENTATION"
for t in $testcases; do
   sed -i 's/whizard/internal/g' $t.json
   ../phs.x $t.json
   if [ -f compare.gpu_cpu ]; then
     mv compare.gpu_cpu compare_${t}_${d}.gpu_cpu
   fi

done

grep Failed\ events *.gpu > summary.log
