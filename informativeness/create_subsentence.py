import pickle
import nltk
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-p', required=False, help='select constituency parser from [base, large]', type=str, choices=['base', 'large'], default='base')
parser.add_argument('-pp', required=False, help='select constituency parser pipeline from [sm, md, lg]', type=str, choices=['sm', 'md', 'lg'], default='lg')
parser.add_argument('-d', required=False, help='select dataset from [wiki1m, STS12, STS13, STS14, STS15, STS16, STS-B, SICK-R, nli, quora, simplewiki, specter, covid]', type=str, choices=['wiki1m', 'STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STS-B', 'SICK-R', 'nli', 'quora', 'simplewiki', 'specter', 'covid', 'huffpost'], default='wiki1m')

DATASET = parser.parse_args().d
PARSER = 'benepar_en3' if parser.parse_args().p == 'base' else 'benepar_en3_large'
PIPELINE = f'en_core_web_{parser.parse_args().pp}'
N = ''
print(DATASET)

def vis_tree(doc):
    try:
        doc._._constituent_data
        tree_type = 'cst'
    except:
        tree_type = 'dpd'
    print ("{:<15} | {:<8} | {:<15} | {:<20}".format('Token','Relation','Parent', 'Children'))
    print ("-" * 70)
    for token in doc:
        if tree_type == 'dpd':
            info = str(token.dep_)
        elif tree_type == 'cst':
            info = str(token._.parse_string[1:].split()[0])
        # Print the token, dependency nature, head and all dependents of the token
        print ("{:<15} | {:<8} | {:<15} | {:<20}"
            .format(str(token.text), info, str(token.head.text), str([child for child in token.children])))

# # Use displayCy to visualize the dependency 
# displacy.render(doc, style='dep', jupyter=True, options={'distance': 120})

# ##### get subsentence from dependency tree
# with open(f'data/{DATASET}_tree_dpd.pickle', 'rb') as f:
#     list_tree = pickle.load(f)

# def walk_tree_dpd(node, depth):
#     if node.n_lefts + node.n_rights > 0:
#         return max(walk_tree_dpd(child, depth + 1) for child in node.children)
#     else:
#         return depth

# def get_subsentence_from_dpd_tree(node, max_depth, it=0):
#     result = []
#     if it == max_depth:
#         if node.dep_ not in list2remove:
#             return [node.text]
#         else:
#             return []
#     it += 1
#     if node.n_lefts:
#         for child in node.lefts:
#             result_temp = get_subsentence_from_dpd_tree(child, max_depth, it)
#             result.extend(result_temp)
#     result.append(node.text)
#     if node.n_rights:
#         for child in node.rights:
#             result_temp = get_subsentence_from_dpd_tree(child, max_depth, it)
#             result.extend(result_temp)
#     return result

# doc = list_tree[50000]
# vis_tree(doc)
# node = list(doc.sents)[0].root

# depth = walk_tree_dpd(node, 0)
# for i in range(1, depth + 1):
#     ' '.join(get_subsentence_from_dpd_tree(node, max_depth=i))

##### get subsentence from constituency tree
list2find = ['ROOT', 'nsubj', 'aux', 'dobj', '']
list2remove = ['dative', 'prep', 'ccomp']

with open(f'data/{DATASET}_tree_cst_{PIPELINE[12:]}{PARSER[11:]}.pickle', 'rb') as f:
    list_tree = pickle.load(f)

doc = list_tree[100]
vis_tree(doc)
node = list(doc.sents)[0]
tree = nltk.tree.Tree.fromstring(node._.parse_string)
tree.pprint()

# transform to nltk tree
def to_nltk_tree(doc):
    try:
        return nltk.tree.Tree.fromstring(list(doc.sents)[0]._.parse_string)
    except:
        return nltk.tree.Tree('S', [])

list_tree_nltk = []
for doc in tqdm(list_tree):
    tree = to_nltk_tree(doc)
    list_tree_nltk.append(tree)

with open(f'data/{DATASET}_tree_cst_{PIPELINE[12:]}{PARSER[11:]}_nltk.pickle', 'wb') as f:
    pickle.dump(list_tree_nltk, f)

with open(f'data/{DATASET}_tree_cst_{PIPELINE[12:]}{PARSER[11:]}_nltk.pickle', 'rb') as f:
    list_tree_nltk = pickle.load(f)

# make subsentence
list2find = ['S', 'NP', 'VP']
list2remove = ['PP', 'ADVP', 'SBAR'] # removed JJ

def walk_tree_cst(node, depth):
    if list(node._.children):
        return max([walk_tree_cst(child, depth + 1) for child in node._.children])
    else:
        return depth

def get_subsentence_from_cst_tree(node, max_depth, it=0):
    if it == max_depth:
        if not isinstance(node, str) and node.label() in list2remove:
            node.clear()
        else:
            pass
    else:
        it += 1
        for child in node:
            get_subsentence_from_cst_tree(child, max_depth, it)

list_subsentence = []
tree = list_tree_nltk[1]
for tree in tqdm(list_tree_nltk):
    if len(tree):
        depth = tree.height()
        list_subsentence_temp = []
        for i in range(depth, 0, -1):
            get_subsentence_from_cst_tree(tree, max_depth=i)
            subsentence_temp = ' '.join(tree.leaves())
            subsentence_temp = subsentence_temp.replace('-LRB-', '(').replace('-RRB-', ')')
            if subsentence_temp not in list_subsentence_temp:
                list_subsentence_temp.append(subsentence_temp)
        list_subsentence.append(list_subsentence_temp)
    else:
        list_subsentence.append(['.'])
list_subsentence[100]
len(list_subsentence)

with open(f'data/{DATASET}_tree_cst_{PIPELINE[12:]}{PARSER[11:]}_subsentence.pickle', 'wb') as f:
    pickle.dump(list_subsentence, f)
