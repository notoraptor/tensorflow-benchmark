#!/usr/bin/env bash

#### Chargment du conteneur docker fourni par NVIDIA (en ligne sur NVIDIA GPU cloud)
nvidia-docker run -it --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 --rm nvcr.io/nvidia/tensorflow:17.12

## Récupération et exécution du script (dans docker)
git clone https://github.com/notoraptor/tensorflow-benchmark.git
cd tensorflow-benchmark

# Exemple d'exécution du benchmark:
# On teste float16 et float32 en même temps.
# On utilise 4096 entrées dans 1 seul run sur 1 seul GPU pendant 2000 étapes.
# Le tenseur d'entrée contient 4096 * 2048 valeurs ((nbatch/nrun) * nin).
# Le tenseur de sortie contiendra 4096 * 2048 valeurs ((nbatch/nrun) * nout).
# Le modèle utilise 100 couches (layers) contenant chacune 2048 neurones (layer-neurons).
# Chaque couche est un simple matmul: tenseur_entree * tenseur_thetas + tenseur_biais.
# Dimensions des tenseurs:
# - tenseur_entree: (nbatch/nrun) * n_parametres
# - tenseur_thetas: n_parametres * layers
# - tenseur_biais: (nbatch/nrun) * layers
python benchmark/benchmark_matmul.py --dtype float16 --dtype float32 --nbatch 4096 --nin 2048 --nout 2048 --nsteps 2000 --nruns 1 --ngpus 1 --layers 100 --layer-neurons 2048

# Même exemple avec nvprof et redirection dans des fichiers de sortie:
nvprof python benchmark/benchmark_matmul.py --dtype float16 --dtype float32 --nbatch 4096 --nin 2048 --nout 2048 --nsteps 2000 --nruns 1 --ngpus 1 --layers 100 --layer-neurons 2048 > out.log 2> out.err &

# Pour obtenir de l'aide pour utiliser le script:
python benchmark/benchmark_matmul.py --help
