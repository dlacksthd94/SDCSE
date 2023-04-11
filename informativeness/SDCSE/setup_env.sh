#!/bin/bash

source ~/anaconda3/etc/profile.d/conda.sh
conda deactivate
conda env list

conda env remove -n sdcse
conda env list

conda create -n sdcse python=3.7 -y
conda activate sdcse
conda env list

pip install torch==1.7.1+cu110 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
