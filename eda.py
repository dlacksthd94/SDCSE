import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import scale
from tqdm import tqdm

# load dataset
with open('../SimCSE/wiki1m_for_simcse.txt') as f:
    list_text = f.readlines()

# preprocessing
    for i in tqdm(range(len(list_text))):
        list_text[i] = list_text[i].strip()

# load embedding
with open(f'wiki1m_for_simcse_embedding.pickle', 'rb') as f:
    list_embedding = pickle.load(f)
    embeddings = np.array(list_embedding)

# load tree depth
with open(f'wiki1m_for_simcse_tree_depth.pickle', 'rb') as f:
    list_depth = pickle.load(f)
    tree_depths = np.array(list_depth)

df = pd.DataFrame(embeddings)
df_summary = df.describe()

# embedding distribution
embedding_dist = embeddings[:, 100]
plt.hist(embedding_dist, bins=50)
plt.title('embedding distribution')
plt.savefig('wiki1m_for_simcse_embedding_dist.png')
plt.clf()

# embedding vector length
embedding_length = (embeddings ** 2).sum(axis=1) ** 0.5
plt.hist(embedding_length, bins=50)
plt.title('embedding vector length')
plt.savefig('wiki1m_for_simcse_embedding_length.png')
plt.clf()

df_all = pd.DataFrame({'text': list_text, 'vector_norm': embedding_length, 'tree_depth': tree_depths})
df_all['text_len'] = df_all['text'].str.split().str.len()
df_all.sample(100000).sort_values('vector_norm', ascending=False)
df_all.corr()

# (np.divide(embeddings, ((embeddings ** 2).sum(axis=1) ** 0.5).reshape(embeddings.shape[0], 1)) ** 2).sum(axis=1) ** 0.5