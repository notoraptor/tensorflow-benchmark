#!/usr/bin/env bash

set -e

export TIMESTAMP=`date +%Y-%m-%d-%Hh-%Mmin-%Ssec`

nvprof python benchmark/benchmark_matmul.py --dtype float16 --dtype float32 --nbatch 4096 --nin 2048 --nout 2048 --nsteps 2000 --nruns 1 --ngpus 1 --layers 100 --layer-neurons 2048 > out-${TIMESTAMP}.log 2> out-${TIMESTAMP}.err

mail -s "job done" stevenbocco@gmail.com < /dev/null
