import torch
from torch import nn;
from torch.nn import functional as trnf


def chunk_shuf_(protos,rchunk,cchunk):
    N,C,H,W=protos.shape;
    spro = protos.reshape(N,C,rchunk,H//rchunk,cchunk,W//cchunk).permute(2,4,0,1,3,5);
    spro=spro.reshape(rchunk*cchunk,N,C,H//rchunk,W//cchunk);
    idxs=torch.stack([torch.randperm(N) for i in range(rchunk*cchunk)]);
    spro=torch.stack([spro[i][idxs[i]] for i in range(rchunk*cchunk)],0);
    spro=spro.reshape(rchunk,cchunk,N,C,H//rchunk,W//cchunk).permute(2,3,0,4,1,5).reshape(N,C,H,W).contiguous();
    return spro,idxs


def chunk_shuf(protos,rchunk,cchunk):
    with torch.no_grad():
        N,C,H,W=protos.shape;
        if(H%rchunk==0 and W%cchunk==0):
            return chunk_shuf_(protos,rchunk,cchunk);
        if(H%rchunk):
            HH=H//rchunk*rchunk+rchunk;
        if (W % cchunk):
            WW =W // cchunk * cchunk + cchunk;
        protos_=trnf.interpolate(protos,[HH,WW]);
        spro_, idxs=chunk_shuf_(protos_,rchunk,cchunk);
        spro=trnf.interpolate(spro_,[H,W])
    return spro,idxs


class neko_rand_shuf(nn.Module):
    def __init__(this,rchunk,cchunk):
        this.rchunk=rchunk;
        this.cchunk=cchunk;
        super(neko_rand_shuf, this).__init__();
    def forward(this,protos):
        return chunk_shuf(protos,this.rchunk,this.cchunk);
