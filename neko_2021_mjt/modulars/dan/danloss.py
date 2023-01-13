
from torch import nn
import torch
from torch.nn import functional as trnf
try:
    import pylcs
except:
    pylcs=None;
    print("no pylcs!, some loss (trident net) won't work!");

class osdanloss_clsemb(nn.Module):
    def __init__(this, cfgs):
        super(osdanloss_clsemb, this).__init__();
        this.setuploss(cfgs);

    def setuploss(this, cfgs):
        this.criterion_CE = nn.CrossEntropyLoss();
        # this.aceloss=
        this.wcls = cfgs["wcls"];
        this.wemb = cfgs["wemb"];
        this.wrej=cfgs["wrej"];
        this.reduction=cfgs["reduction"];

    def forward(this, proto, outcls, label_flatten):
        if(this.wemb>0):
            proto_loss = trnf.relu(proto[1:].matmul(proto[1:].T) - 0.14).mean();
        else:
            proto_loss=torch.tensor(0).float().to(outcls.device);
        w = torch.ones_like(torch.ones(outcls.shape[-1])).to(outcls.device).float();

        # w[-1] = 0.1;
        clsloss = trnf.cross_entropy(outcls, label_flatten, w,ignore_index=-1);
        loss =  clsloss * this.wcls  + this.wemb * proto_loss;
        terms = {
            "total": loss.detach().item(),
            "main": clsloss.detach().item(),
            "emb": proto_loss.detach().item(),
        }
        return loss, terms

class clsloss(nn.Module):
    def __init__(this, cfgs):
        super(clsloss, this).__init__();
        this.setuploss(cfgs);

    def setuploss(this, cfgs):
        this.criterion_CE = nn.CrossEntropyLoss(ignore_index=-1);
        # this.aceloss=
        this.wcls = cfgs["wcls"];

    def forward(this, proto, outcls, label_flatten):
        # w[-1] = 0.1;
        clsloss = this.criterion_CE(outcls, label_flatten);
        terms = {
            "clsloss": clsloss.detach().item(),
        }
        return clsloss, terms
