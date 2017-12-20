#!/usr/bin/env bash

#### (Anciennement) Compilation et installation d'un conteneur docker à partir du dépôt Tensorflow
## Installation du conteneur docker
git clone https://github.com/tensorflow/tensorflow.git
cd tensorflow/tensorflow/tools/docker
nvidia-docker build -t tensorflow:cuda9 -f Dockerfile.devel-gpu-cuda9-cudnn7 .
## Chargement du conteneur docker
nvidia-docker run -it tensorflow:cuda9

#### (Nouveau) Chargment d'un conteneur docker fourni par NVIDIA (téléchargeable en ligne sur NVIDIA GPU cloud)
nvidia-docker run -it --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 --rm nvcr.io/nvidia/tensorflow:17.12

## Récupération et exécution du script (dans docker)
git clone https://github.com/notoraptor/tensorflow-benchmark.git
cd tensorflow-benchmark
# Le benchmark va traiter <nbatch> images de dimensions <nin> * <nin> pour <nout> classes de sortie.
# Les <nbatch> images peuvent être réparties dans <nruns> exécutions parallèles, chacune sur un GPU.
# Chaque exécution calcule <nsteps> étapes.
python benchmark/benchmark_deep.py --dtype float32 --nbatch 1000 --nin 64 --nout 10 --nsteps 1000 --nruns 2 --ngpus 2
# Exemple pour tester plusieurs dtypes en même temps
# (1 benchmark par dtype, les temps d'exécution sont tous affichés à la fin):
python benchmark/benchmark_deep.py --dtype float16 --dtype float32 ...  # autres paramètres

# Pour obtenir plus d'aide:
python benchmark/benchmark_deep.py --help
