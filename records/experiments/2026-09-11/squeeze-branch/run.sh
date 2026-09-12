set -u
cd /home/yspark/projects/ICRA
PY="nice -n 15 env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MUJOCO_GL=egl third_party/dexonomy/.venv/bin/python"
LIMIT=1800
STATUS=out/squeeze-branch/run-status.tsv
printf "name\tarm\texit_code\ttimed_out\tseconds\n" > $STATUS
one () {  # name data episode candidates arm
  start=$(date +%s)
  timeout $LIMIT $PY scripts/squeeze_branch_compare.py --data "$2" --episode "$3" \
      --candidates "$4" --arm "$5" --out "out/squeeze-branch/$1_$5.json"
  code=$?
  end=$(date +%s)
  timedout=no; [ $code -eq 124 ] && timedout=yes
  printf "%s\t%s\t%d\t%s\t%d\n" "$1" "$5" "$code" "$timedout" "$((end-start))" >> $STATUS
  echo "   $1 arm $5: exit $code, timed_out=$timedout, $((end-start))s"
}
BOWL=out/bowl-scale/s008o4/executed
BOWLC=out/bowl-scale/manifests/s008o4-execute.json
CAN=out/multiobject/can/executed-all
CANC=out/multiobject/manifests/can-distinct.json
for arm in A B; do
  one bowl0000 $BOWL teacher_0000_yaw_000 $BOWLC $arm
  one bowl0006 $BOWL teacher_0006_yaw_000 $BOWLC $arm
  one can      $CAN  teacher_0008_yaw_000 $CANC  $arm
done
echo SQUEEZE6DONE
