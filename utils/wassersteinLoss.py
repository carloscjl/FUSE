import torch
import torch.nn as nn
import torchvision.datasets as dsets
from torchvision import datasets, transforms
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
import pandas as pd



def moco_loss(query, keys, queue, temperature=0.07):
    """
    query: Tensor of shape [batch_size, feature_dim], representing query embeddings.
    keys: Tensor of shape [batch_size, n, feature_dim], representing n positive keys for each query.
    queue: Tensor of shape [K, feature_dim], representing the queue of negative keys.
    temperature: scalar, temperature hyperparameter for the InfoNCE loss.
    
    Returns: loss, scalar tensor.
    """
    batch_size, feature_dim = query.shape
    n = keys.shape[1]
    K = queue.size(0)
 
    # Normalize embeddings
    query = F.normalize(query, dim=1)
    keys = F.normalize(keys, dim=2)
    queue = F.normalize(queue, dim=1)
 
    # Compute positive similarities: [batch_size, n]
    pos_sim = torch.einsum('ik,ijk->ij', [query, keys]) / temperature
    
    # First reshape the pos_sim
    pos_sim_reshaped = pos_sim.view(batch_size * n, 1)
 
    # Compute negative similarities: [batch_size, K]
    neg_sim = torch.mm(query, queue.t()) / temperature
    # Then repeat the neg_sim to get :(batch_size*n,K)
    neg_sim_repeated = neg_sim.repeat(1, n).view(batch_size * n, K)
 
    # Construct the logits: [batch_size * n, 1 + K]
    logits = torch.cat([pos_sim_reshaped, neg_sim_repeated], dim=1)
    
    # Targets: [batch_size*n,]
    targets = torch.arange(0, n).long().to(query.device).repeat(batch_size).view(-1)
 
    loss = F.cross_entropy(logits, targets)
 
    return loss

def sink(M, reg = 1, numItermax=1000, stopThr=1e-9, cuda = True):

    # we assume that no distances are null except those of the diagonal of
    # distances

    if cuda:
        a = Variable(torch.ones((M.size()[0],)) / M.size()[0]).cuda()
        b = Variable(torch.ones((M.size()[1],)) / M.size()[1]).cuda()
    else:
        a = Variable(torch.ones((M.size()[0],)) / M.size()[0])
        b = Variable(torch.ones((M.size()[1],)) / M.size()[1])

    # init data
    Nini = len(a)
    Nfin = len(b)

    if cuda:
        u = Variable(torch.ones(Nini) / Nini).cuda()
        v = Variable(torch.ones(Nfin) / Nfin).cuda()
    else:
        u = Variable(torch.ones(Nini) / Nini)
        v = Variable(torch.ones(Nfin) / Nfin)

    # print(reg)

    K = torch.exp(-M / reg)
    # print(np.min(K))

    Kp = (1 / a).view(-1, 1) * K
    cpt = 0
    err = 1
    while (err > stopThr and cpt < numItermax):
        uprev = u
        vprev = v
        #print(T(K).size(), u.view(u.size()[0],1).size())
        KtransposeU = K.t().matmul(u)
        v = torch.div(b, KtransposeU)
        u = 1. / Kp.matmul(v)

        if cpt % 10 == 0:
            # we can speed up the process by checking for the error only all
            # the 10th iterations
            transp = u.view(-1, 1) * (K * v)
            err = (torch.sum(transp) - b).norm(1).pow(2).item()


        cpt += 1

    return torch.sum(u.view((-1, 1)) * K * v.view((1, -1)) * M)


def sink_stabilized(M, reg, numItermax=1000, tau=1e2, stopThr=1e-9, warmstart=None, print_period=20, cuda=True):

    if cuda:
        a = Variable(torch.ones((M.size()[0],)) / M.size()[0]).cuda()
        b = Variable(torch.ones((M.size()[1],)) / M.size()[1]).cuda()
    else:
        a = Variable(torch.ones((M.size()[0],)) / M.size()[0])
        b = Variable(torch.ones((M.size()[1],)) / M.size()[1])

    # init data
    na = len(a)
    nb = len(b)

    cpt = 0
    # we assume that no distances are null except those of the diagonal of
    # distances
    if warmstart is None:
        if cuda:
            alpha, beta = Variable(torch.zeros(na)).cuda(), Variable(torch.zeros(nb)).cuda()
        else:
            alpha, beta = Variable(torch.zeros(na)), Variable(torch.zeros(nb))
    else:
        alpha, beta = warmstart

    if cuda:
        u, v = Variable(torch.ones(na) / na).cuda(), Variable(torch.ones(nb) / nb).cuda()
    else:
        u, v = Variable(torch.ones(na) / na), Variable(torch.ones(nb) / nb)

    def get_K(alpha, beta):
        return torch.exp(-(M - alpha.view((na, 1)) - beta.view((1, nb))) / reg)

    def get_Gamma(alpha, beta, u, v):
        return torch.exp(-(M - alpha.view((na, 1)) - beta.view((1, nb))) / reg + torch.log(u.view((na, 1))) + torch.log(v.view((1, nb))))

    # print(np.min(K))

    K = get_K(alpha, beta)
    transp = K
    loop = 1
    cpt = 0
    err = 1
    while loop:

        uprev = u
        vprev = v

        # sinkhorn update
        v = torch.div(b, (K.t().matmul(u) + 1e-16))
        u = torch.div(a, (K.matmul(v) + 1e-16))

        # remove numerical problems and store them in K
        if torch.max(torch.abs(u)).data[0] > tau or torch.max(torch.abs(v)).data[0] > tau:
            alpha, beta = alpha + reg * torch.log(u), beta + reg * torch.log(v)

            if cuda:
                u, v = Variable(torch.ones(na) / na).cuda(), Variable(torch.ones(nb) / nb).cuda()
            else:
                u, v = Variable(torch.ones(na) / na), Variable(torch.ones(nb) / nb)

            K = get_K(alpha, beta)

        if cpt % print_period == 0:
            transp = get_Gamma(alpha, beta, u, v)
            err = (torch.sum(transp) - b).norm(1).pow(2).data[0]

        if err <= stopThr:
            loop = False

        if cpt >= numItermax:
            loop = False

        #if np.any(np.isnan(u)) or np.any(np.isnan(v)):
        #    # we have reached the machine precision
        #    # come back to previous solution and quit loop
        #    print('Warning: numerical errors at iteration', cpt)
        #    u = uprev
        #    v = vprev
        #    break

        cpt += 1

    return torch.sum(get_Gamma(alpha, beta, u, v)*M)

def pairwise_distances(x, y, method='l1'):
    n = x.size()[0]
    m = y.size()[0]
    d = x.size()[1]

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    if method == 'l1':
        dist = torch.abs(x - y).sum(2)
    else:
        dist = torch.pow(x - y, 2).sum(2)

    return dist.float()

def cost_matrix(x,y):
    xy_T = torch.matmul(x,y.transpose(0,1))
    x2 = torch.sum(torch.square(x),dim=1,keepdim=True)
    y2 = torch.sum(torch.square(y), dim=1, keepdim=True)
    norm = torch.matmul(torch.sqrt(x2), torch.sqrt(y2).transpose(0,1))
    C = 1 - xy_T/norm
    return C

def dmat(x,y):
    mmp1 = torch.stack([x] * x.size()[0])
    mmp2 = torch.stack([y] * y.size()[0]).transpose(0, 1)
    mm = torch.sum((mmp1 - mmp2) ** 2, 2).squeeze()

    return mm

def sinkhron_loss(x,y,epsilon,n,niter):
    C = cost_matrix(x, y)
    mu = torch.tensor(1.0/n)
    nu = torch.tensor(1.0 / n)

    def M(u,v):
        return (-C + u.expand(u.size(0),u.size(0)) + v.expand(v.size(0),v.size(0)) ) / epsilon
    def lse(A):
        return torch.logsumexp(A, dim=1,keepdim=True)

    u, v = 0. * mu, 0. * nu
    for i in range(niter):
        u = epsilon * (torch.log(mu) - torch.squeeze(lse(M(u, v)))) + u
        v = epsilon * (torch.log(nu) - torch.squeeze(lse(torch.transpose(M(u, v))))) + v
    u_final, v_final = u, v
    pi = torch.exp(M(u_final, v_final))
    cost = torch.sum(pi * C)

    return cost

if __name__=='__main__':
    x = torch.ones(10, 5)
    y = torch.rand((10, 5))
    C = cost_matrix(x, y).cuda()
    loss = sink(C)
    print(loss)
