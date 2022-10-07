import torch
import torch.nn.functional as F

# taken from pytorch in case i need to change something in it eventually
def gumbel_softmax(logits, tau=1, hard=True):
    gumbels = (
        -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_().log()
    )  # ~Gumbel(0,1)
    gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
    y_soft = gumbels.softmax(dim)

    if hard:
        # Straight through.
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft
    return ret



def knn(X, Y, k):
    # X * x N x d
    # Y * x M x d
    dists = (X.unsqueeze(-2) - Y.unsqueeze(-3)).norm(dim=-1)    # * x N x M
    _, indices = dists.topk(k, dim=-1)  # * x N x k
    return indices

    