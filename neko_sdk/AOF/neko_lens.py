from torch import nn;
import torch;
import torch.nn.functional as trnf;


class neko_dense_calc(nn.Module):
    def __init__(this,channel):
        super(neko_dense_calc, this).__init__();
        ## 996-251. I am not on purpose.
        ## The sigmoid perserves topology.
        this.dkern=torch.nn.Sequential(
            nn.Conv2d(int(channel),2,5,1),
            torch.nn.Sigmoid());
    def forward(this,feat,lw,lh):
        if(this.training):
            dmap = this.dkern(feat/(feat.norm(dim=1,keepdim=True)+0.00000009));
        else:
            norm=feat.norm(dim=1,keepdim=True);
            dmap = this.dkern(feat/(norm+0.00000009)*(norm>0.09));
            # During large-batch evaluation, numeric errors seems to get much larger than eps,
            # causing sever performance loss, hence we commence this hot-fix.

        ndmap = trnf.interpolate(dmap, [lh + 2, lw + 2]);
        return ndmap;

class neko_dense_calcnn(nn.Module):
    def __init__(this,channel):
        super(neko_dense_calcnn, this).__init__();
        ## 996-251. I am not on purpose.
        ## The sigmoid perserves topology.
        this.dkern=torch.nn.Sequential(
            nn.Conv2d(int(channel),2,5,1),
            torch.nn.Sigmoid());
    def forward(this,feat,lw,lh):
        dmap = this.dkern(feat);
        ndmap = trnf.interpolate(dmap, [lh + 2, lw + 2]);
        return ndmap;



def neko_dense_norm(ndmap):
    [h__, w__] = ndmap.split([1, 1], 1);
    sumh=torch.sum(h__, dim=2, keepdim=True);
    sumw=torch.sum(w__,dim=3,keepdim=True);
    h_ = h__ / sumh;
    w_ = w__ / sumw;
    h = torch.cumsum(h_, dim=2);
    w = torch.cumsum(w_, dim=3);
    nidx = torch.cat([ w[:, :, 1:-1, 1:-1],h[:, :, 1:-1, 1:-1]], dim=1)* 2 - 1;
    return nidx;

def neko_sample(feat,grid,dw,dh):
    dst = trnf.grid_sample(feat, grid.permute(0, 2, 3, 1),mode="bilinear");
    return trnf.adaptive_avg_pool2d(dst,[dh,dw]);

def neko_dsample(feat,dmap,dw,dh):
    grid = neko_dense_norm(dmap);
    dst = neko_sample(feat, grid, dw, dh);
    return dst
def neko_scale(feat,grid,dw,dh):
    dst = trnf.grid_sample(feat, grid.permute(0, 2, 3, 1),mode="bilinear");
    return trnf.interpolate(dst,[dh,dw],mode="bilinear");

def neko_dscale(feat,dmap,dw,dh):
    grid = neko_dense_norm(dmap);
    dst = neko_scale(feat, grid, dw, dh);
    return dst

def neko_idsample(feat,density,dw,dh):
    rdense=1/density;
    grid = neko_dense_norm(rdense);
    dst = neko_sample(feat, grid, dw, dh);
    return dst;

def neko_dense_norm2(ndmap):
    [h__, w__] = ndmap.split([1, 1], 1);
    sumh=torch.sum(h__, dim=2, keepdim=True);
    sumw=torch.sum(w__,dim=3,keepdim=True);
    h_ = h__ / sumh;
    w_ = w__ / sumw;
    h = torch.cumsum(h_, dim=2);
    w = torch.cumsum(w_, dim=3);
    nidx = torch.cat([ w,h], dim=1)* 2 - 1;
    return nidx;
def neko_sample2(feat,grid):
    dst = trnf.grid_sample(feat, grid.permute(0, 2, 3, 1),mode="bilinear",align_corners=True);
    return dst;

def neko_dsample2(feat,dmap):
    grid = neko_dense_norm(dmap);
    dst = neko_sample2(feat, grid);
    return dst
def neko_idsample2(feat,density):
    rdense=1/density;
    grid = neko_dense_norm(rdense);
    dst = neko_sample2(feat, grid);
    return dst;

def vis_lenses(img,lenses):
    oups=[img];
    for lens in lenses:
        dmap=trnf.interpolate(lens, [img.shape[-2],img.shape[-1]])
        grid=neko_dense_norm(dmap);
        img=neko_sample(img,grid,img.shape[3],img.shape[2])
        oups.append(img);
    return oups;

class neko_lens(nn.Module):
    DENSE=neko_dense_calc
    def __init__(this,channel,pw,ph,hardness=2,dbg=False):
        super(neko_lens, this).__init__();
        this.pw=1;
        this.ph=1;
        this.dbg=dbg;
        this.hardness=hardness;
        this.dkern=this.DENSE(channel)


    def forward(this,feat):
        dw = int(feat.shape[3] / this.pw);
        dh = int(feat.shape[2] / this.ph);
        dmap=this.dkern(feat,dw,dh);
        dmap += 0.0009;  # sigmoid can be clipped to 0!
        dmap[:, :, 1:-1, 1:-1] += this.hardness;
        dst=neko_dsample(feat,dmap,dw,dh);

        if(not this.dbg):
            return dst,dmap.detach();
        else:
            return dst,dmap.detach();
