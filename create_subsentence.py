import pickle

with open(f'wiki1m_for_simcse_tree_dpd.pickle', 'rb') as f:
    list_tree = pickle.load(f)

# get subsentence from dpd tree
list2find = ['ROOT', 'nsubj', 'aux', 'dobj', '']
list2remove = ['dative', 'prep', 'ccomp']

def walk_tree(node, depth):
    if node.n_lefts + node.n_rights > 0:
        return max(walk_tree(child, depth + 1) for child in node.children)
    else:
        return depth

def get_subsentence_from_dpd_tree(node, max_depth, it=0):
    result = []
    if it == max_depth:
        if node.dep_ not in list2remove:
            return [node.text]
        else:
            return []
    it += 1
    if node.n_lefts:
        for child in node.lefts:
            result_temp = get_subsentence_from_dpd_tree(child, max_depth, it)
            result.extend(result_temp)
    result.append(node.text)
    if node.n_rights:
        for child in node.rights:
            result_temp = get_subsentence_from_dpd_tree(child, max_depth, it)
            result.extend(result_temp)
    return result

doc = list_tree[50000]
node = list(doc.sents)[0].root
depth = walk_tree(node, 0)
for i in range(1, depth + 1):
    ' '.join(get_subsentence_from_dpd_tree(node, max_depth=i))

print ("{:<15} | {:<8} | {:<15} | {:<20}".format('Token','Relation','Head', 'Children'))
print ("-" * 70)
for token in doc:
  # Print the token, dependency nature, head and all dependents of the token
  print ("{:<15} | {:<8} | {:<15} | {:<20}"
         .format(str(token.text), str(token.dep_), str(token.head.text), str([child for child in token.children])))

