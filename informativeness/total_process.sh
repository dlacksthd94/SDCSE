#!/bin/sh

########## set up env ##########
bash setup_env.sh

########## git clone ##########
# SimCSE
git clone https://github.com/princeton-nlp/SimCSE.git

# DiffCSE
git clone https://github.com/voidism/DiffCSE.git
# create env for diffcse
source ~/anaconda3/etc/profile.d/conda.sh
conda create -n diffcse python=3.9.5 -y
conda env list
conda activate diffcse
cd DiffCSE
    cd transformers-4.2.1
        pip install .
    cd ..
    pip install torch==1.7.1+cu110 -f https://download.pytorch.org/whl/torch_stable.html # necessary to use GPU
    pip install -r requirements.txt
    pip install dill==0.2.8.2
cd ..
#####
##### In order to prevent torch.distributed parallelism error, you have to manually edit 514~518th lines in diffcse/traners.py by changing 'model.~' -> 'model.module.~'
#####

# PromCSE
git clone https://github.com/YJiangcm/PromCSE.git

# MCSE
git clone https://github.com/uds-lsv/MCSE.git

# ESimCSE, InfoCSE, gsInfoCSE
git clone https://github.com/caskcsg/sentemb.git
mv sentemb/* .
rm -rf sentemb

# InforMin-CL
git clone https://github.com/ShaobinChen-AH/InforMin-CL.git

# RankCSE
git clone https://github.com/perceptiveshawty/RankCSE.git

# PeerCL
git clone https://github.com/qiyuw/PeerCL.git

# MixCSE
git clone https://github.com/BDBC-KG-NLP/MixCSE_AAAI2022.git
mv MixCSE_AAAI2022 MixCSE
#####
##### In order to run train.py normally, toggle line comment on/off in line 350, 351, 353.
#####

# SCD
git clone https://github.com/SAP-samples/acl2022-self-contrastive-decorrelation.git
mv acl2022-self-contrastive-decorrelation SCD

########## download dataset ##########
# SimCSE
cd SimCSE
    cd data
        bash download_wiki.sh
        bash download_nli.sh
    cd ..
    cd SentEval/data/downstream/
        bash download_dataset.sh
    cd ../../..
cd ..

# DiffCSE
cd DiffCSE
    cd data
        bash download_wiki.sh
    cd ..
    cd SentEval/data/downstream/
        bash download_dataset.sh
    cd ../../..
cd ..

# PromCSE
cd PromCSE
    cd data
        bash download_wiki.sh
        bash download_nli.sh
    cd ..
    cd SentEval/data/downstream/
        bash download_dataset.sh
    cd ../../..
cd ..

# MCSE
cd MCSE
    cd data
        # wiki1m dataset
        wget https://huggingface.co/datasets/princeton-nlp/datasets-for-simcse/resolve/main/wiki1m_for_simcse.txt
        # flickr dataset
        FILEID="$(cut -d'/' -f6 <<<'https://drive.google.com/file/d/1TBNIM9-zL-wXb2kH8YtuYPig2oYInSSm/view?usp=sharing')"
        FILENAME=flickr_random_captions.txt
        wget --no-check-certificate "https://docs.google.com/uc?export=download&id=${FILEID}" -O ${FILENAME}
        FILEID="$(cut -d'/' -f6 <<<'https://drive.google.com/file/d/10x7Kf5ZD406gxcxONNn_CBpoaxij6Jv0/view')"
        FILENAME=flickr_resnet.hdf5
        wget --load-cookies ~/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies ~/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=${FILEID}' -O- | sed -rn 's/.confirm=([0-9A-Za-z_]+)./\1\n/p')&id=${FILEID}" -O ${FILENAME} && rm -rf ~/cookies.txt
        # CoCo dataset
        FILEID="$(cut -d'/' -f6 <<<'https://drive.google.com/file/d/1wKmsDtvjtWYSCxnJp_BHGxO3HfcPdc6c/view')"
        FILENAME=coco_random_captions.txt
        wget --no-check-certificate "https://docs.google.com/uc?export=download&id=${FILEID}" -O ${FILENAME}
        FILEID="$(cut -d'/' -f6 <<<'https://drive.google.com/file/d/1UR0XIrh9b9W7ydjQSx4MwVFnWc5OwV7p/view')"
        FILENAME=coco_resnet.hdf5
        wget --load-cookies ~/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies ~/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=${FILEID}' -O- | sed -rn 's/.confirm=([0-9A-Za-z_]+)./\1\n/p')&id=${FILEID}" -O ${FILENAME} && rm -rf ~/cookies.txt
    cd ..
    cd SentEval/data/downstream/
        bash download_dataset.sh
    cd ../../..
cd ..

# ESimCSE
cd ESimCSE
    cd data
        bash download_wiki.sh
    cd ..
    cd SentEval/data/downstream/
        bash download_dataset.sh
    cd ../../..
cd ..

# InfoCSE
cd InfoCSE
    cd data
        bash download_wiki.sh
    cd ..
    cd SentEval/data/downstream/
        bash download_dataset.sh
    cd ../../..
cd ..

# gsInfoNCE
cd gsInfoNCE
    cd data
        bash download_wiki.sh
    cd ..
    cd SentEval/data/downstream/
        bash download_dataset.sh
    cd ../../..
cd ..

# InforMin-CL
cd InforMin-CL
    cd data
        bash download_wiki.sh
    cd ..
    unzip SentEval
    cd SentEval/data/downstream/
        bash download_dataset.sh
    cd ../../..
cd ..

# RankCSE
cd RankCSE
    mkdir data
    cd data
        cp ../../SimCSE/data/download_wiki.sh .
        bash download_wiki.sh
    cd ..
    cd SentEval/data/downstream/
        bash download_dataset.sh
    cd ../../..
    mkdir result
cd ..

# PeerCL
cd PeerCL
    cd data
        bash download_wiki.sh
    cd ..
    mkdir -p SentEval/data/downstream/
    mv SentEval/download_dataset.sh SentEval/data/downstream/
    cd SentEval/data/downstream/
        bash download_dataset.sh
    cd ../../..
cd ..

# MixCSE
cd MixCSE
    mkdir -p data/text
    cd data
        cp ../../SimCSE/data/download_wiki.sh .
        bash download_wiki.sh
    cd ..
    cp -r ../SimCSE/SentEval .
    cd SentEval/data/downstream/
        bash download_dataset.sh
    cd ../../..
    mkdir result
cd ..

# SCD
cd SCD
    cd data
        bash download_wiki.sh
    cd ..
    cd SentEval/data/downstream/
        bash download_dataset.sh
    cd ../../..
cd ..
