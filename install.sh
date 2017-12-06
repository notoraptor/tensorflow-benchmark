## Installation du conteneur docker
git clone https://github.com/tensorflow/tensorflow.git
cd tensorflow/tensorflow/tools/docker
nvidia-docker build -t tensorflow:cuda9 -f Dockerfile.devel-gpu-cuda9-cudnn7 .

## Chargement du conteneur docker
nvidia-docker run -it tensorflow:cuda9

## Récupération et exécution du script
git clone https://github.com/notoraptor/tensorflow-benchmark.git
cd tensorflow-benchmark
python benchmark/benchmark_deep.py
