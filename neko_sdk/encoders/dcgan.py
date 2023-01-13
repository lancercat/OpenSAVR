
from torch import nn;
import torch;

class DCGAN_reconstruction_network_insnorm(nn.Module):
    @classmethod
    def setnorm(cls,ch):
        return torch.nn.InstanceNorm2d(ch)
    def __init__(this,ifc,ngf=64,nc=1):
        super(DCGAN_reconstruction_network_insnorm, this).__init__()
        this.rconv1=nn.ConvTranspose2d(ifc, ngf * 8, 2, 1, 0, bias=False);
        this.bn1=this.setnorm(ngf*8);
        this.relu=nn.ReLU(True);
        # state size. (ngf*8) x 4 x 4
        this.rconv2=nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False);
        this.bn2=this.setnorm(ngf * 4);
        # state size. (ngf*4) x 8 x 8
        this.rconv3=nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False);
        this.bn3=this.setnorm(ngf * 2);
        # state size. (ngf*2) x 16 x 16
        this.rconv4=nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False);
        this.bn4=this.setnorm(ngf);
        # state size. (ngf) x 32 x 32
        this.rconv5=nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False);
        # avoid accident name crash
        this.neko_module_list=[this.rconv1,this.bn1,this.relu,
                      this.rconv2,this.bn2,this.relu,
                      this.rconv3,this.bn3,this.relu,
                      this.rconv4,this.bn4,this.relu,
                      this.rconv5,
                      ];
        pass;

    def forward(this,protos):
        outs=[];
        t=protos;
        for m in this.neko_module_list:
            t=m(t);
            outs.append(t);
        return t;
