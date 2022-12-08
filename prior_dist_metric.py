# %%
import torch

# %%
# Utilities for loss implementation.


def get_logsumexp_loss(states, temperature):
  # scores = torch.matmul(states, states.T)  # (bsz, bsz)
  scores = torch.matmul(states, states.T)  # (bsz, bsz)
  # bias = torch.log(torch.cast(torch.shape(states)[1], torch.float32))  # a constant
  bias = torch.log(torch.tensor(20*states.shape[0], dtype=torch.float32))  # a constant
  return  torch.mean(torch.logsumexp(scores / temperature, 1) - bias).item()


# def sort(x):
#   """Returns the matrix x where each row is sorted (ascending)."""
#   xshape = x.shape
#   rank = torch.sum(
#       torch.cast(torch.expand_dims(x, 2) > torch.expand_dims(x, 1), torch.int32), axis=2)
#   rank_inv = torch.einsum(
#       'dbc,c->db',
#       torch.transpose(torch.cast(torch.one_hot(rank, xshape[1]), torch.float32), [0, 2, 1]),
#       torch.range(xshape[1], dtype='float32'))  # (dim, bsz)
#   x = torch.gather(x, torch.cast(rank_inv, torch.int32), axis=-1, batch_dims=-1)
#   return x


def get_swd_loss(states, rand_w, prior='normal', stddev=1., hidden_norm=True):
  states_shape = states.shape
  states = torch.matmul(states, rand_w)
  # states_t = sort(torch.transpose(states))  # (dim, bsz)
  states_t, _ = torch.sort(states.T)

  if prior == 'normal':
    # states_prior = torch.randn(states_shape, mean=0, stddev=stddev)
    states_prior = torch.tensor(torch.randn(states_shape))
  elif prior == 'uniform':
    # states_prior = torch.rand(states_shape, -stddev, stddev)
    states_prior = torch.tensor(torch.rand(states_shape))

  else:
    raise ValueError('Unknown prior {}'.format(prior))
  if hidden_norm:
    # states_prior = torch.nn.functional.normalize(states_prior, -1)
    states_prior = states_prior/states_prior.norm(dim=0, keepdim=True)
  states_prior = torch.matmul(states_prior, rand_w)
  # states_prior_t = sort(torch.transpose(states_prior))  # (dim, bsz)
  states_prior_t, _ = torch.sort(states_prior.T)  # (dim, bsz) 
  
  return (states - states_prior).norm(p=2, dim=1).pow(2).mean().item()
  # return torch.mean((states_prior_t - states_t)**2).item()
     

# %%
def generalized_contrastive_loss(
    hidden1,
    hidden2,
    lambda_weight=1.0,
    temperature=1.0,
    dist='normal',
    hidden_norm=True,
    loss_scaling=1.0):
  """Generalized contrastive loss.

  Both hidden1 and hidden2 should have shape of (n, d).

  Configurations to get following losses:
  * decoupled NT-Xent loss: set dist='logsumexp', hidden_norm=True
  * SWD with normal distribution: set dist='normal', hidden_norm=False
  * SWD with uniform hypersphere: set dist='normal', hidden_norm=True
  * SWD with uniform hypercube: set dist='uniform', hidden_norm=False
  """
  hidden_dim = hidden1.shape[-1]  # get hidden dimension
  if hidden_norm:
    # hidden1 = torch.nn.functional.normalize(hidden1, -1)
    # hidden2 = torch.nn.functional.normalize(hidden2, -1)
    hidden1 = hidden1/hidden1.norm(dim=0, keepdim=True)
    hidden2 = hidden2/hidden2.norm(dim=0, keepdim=True)
  # loss_align = torch.mean((hidden1 - hidden2)**2) / 2.
  loss_align = (hidden1-hidden2).norm(p=2, dim=1).pow(2).mean().item()
  # hiddens = torch.concat([hidden1, hidden2], 0)
  hiddens = hidden1

  if dist == 'logsumexp':
    loss_dist_match = get_logsumexp_loss(hiddens, temperature)
  else:
    # initializer = torch.keras.initializers.Orthogonal()
    # rand_w = initializer(torch.tensor(hidden_dim, hidden_dim))
    rand_w = torch.nn.init.orthogonal(torch.rand((hidden_dim, hidden_dim)))
    loss_dist_match = get_swd_loss(hiddens, rand_w,
                            prior=dist,
                            hidden_norm=hidden_norm)
  return loss_align, loss_dist_match
  return loss_scaling * (loss_align + lambda_weight * loss_dist_match)


# %%
prior_dist_list = ['logsumexp', 'normal', 'uniform']
model_list = ['sbert', 'simcse', 'diffcse', 'promcse']
data_list = ['quora', 'simplewiki', 'specter', 'stack']
# x = torch.load('embeddings/quora/simcse/query.pt')
# y = torch.load('embeddings/quora/simcse/pos.pt')

# generalized_contrastive_loss(x, y, hidden_norm=False, dist='logsumexp')

# %%
for data_name in data_list:
    print('-'*36)
    print(data_name)
    for model_name in model_list:
        print(model_name)
        x = torch.tensor(torch.load(f'embeddings/{data_name}/{model_name}/query.pt'))
        y = torch.tensor(torch.load(f'embeddings/{data_name}/{model_name}/pos.pt'))
        
        for prior_dist in prior_dist_list:
            align, unif = generalized_contrastive_loss(x, y, hidden_norm=True, dist=prior_dist)
            print(f'{prior_dist}: Alignment:{align}, Distribution:{unif}   (normalized=True)')
            align, unif = generalized_contrastive_loss(x, y, hidden_norm=False, dist=prior_dist)
            print(f'{prior_dist}: Alignment:{align}, Distribution:{unif}   (normalized=False)')

# %%



