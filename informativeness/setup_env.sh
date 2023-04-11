source ~/anaconda3/etc/profile.d/conda.sh

ENV_NAME="nlp_project"

if [ $ENV_NAME == "nlp_project" ]; then
conda deactivate
conda env list
conda env remove -n ${ENV_NAME}
conda env list

conda create -n ${ENV_NAME} python==3.9.5 -y
conda activate ${ENV_NAME}
conda env list

pip cache purge
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 -f https://download.pytorch.org/whl/torch_stable.html # necessary to use GPU
pip install -r requirements.txt # necessary due to dependency between packages
pip install simcse
pip install matplotlib
pip install numpy==1.21
pip install jsonlines
pip install dill==0.2.8.2 # necessary to prevent error
pip install seaborn

elif [ $ENV_NAME == "spacy" ]; then
conda deactivate
conda env list
conda env remove -n ${ENV_NAME}
conda env list

conda create -n ${ENV_NAME} python==3.6 -y
conda activate ${ENV_NAME}
conda env list

pip install spacy
pip install benepar
python -m spacy download en_core_web_sm
python -m spacy download en_core_web_md
python -m spacy download en_core_web_lg
pip install torch==1.10.1+cu102 torchvision==0.11.2+cu102 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu102/torch_stable.html
pip install nltk
pip install jsonlines
pip install pandas

fi