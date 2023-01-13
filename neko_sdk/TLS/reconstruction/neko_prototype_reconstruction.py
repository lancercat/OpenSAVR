import cv2
import numpy as np
from neko_sdk.encoders.dcgan import DCGAN_reconstruction_network_insnorm
# balanced tanh l2 loss.
from torch import nn;
import torch;
from torch.nn import functional as trnf
from neko_sdk.neko_score_merging import scatter_cvt;
from neko_sdk.loss_functions.balanced_l2 import neko_masked_l2

class neko_dcgan_reconstructionN_insnorm(nn.Module):
    def __init__(this, psize, gsize=64, expf=1, och=1, scale=1.5, bias=0):
        super(neko_dcgan_reconstructionN_insnorm, this).__init__();
        this.scale = scale;
        this.bias = bias;

        this.recon = DCGAN_reconstruction_network_insnorm(psize, gsize, och * expf * expf);
        if (expf > 1):
            this.reconpost = torch.nn.PixelShuffle(expf);
        else:
            this.reconpost = None;
        pass;

    def forward(this, protos):
        raw = this.recon(protos);
        if (this.reconpost is not None):
            raw = this.reconpost(raw);
        return trnf.tanh(raw) * this.scale + this.bias;

class neko_masked_l2_loss(nn.Module):
    def __init__(this,speed=1):
        super(neko_masked_l2_loss,this).__init__();
        this.speed=speed;

    def debug(this,raw,dbgkey):
        raww=(raw[10:20].reshape(640,64).detach()*127.5+128).clip(0,255).detach().cpu().numpy().astype(np.uint8);
        cv2.namedWindow("preview"+str(dbgkey));
        cv2.imshow("preview"+str(dbgkey),raww);

        cv2.waitKey(0);

    def forward(this,raw,gt,dbgkey=None):
        if(dbgkey):
            this.debug(raw,dbgkey);
        with torch.no_grad():
            tar=gt.to(raw.device);
            if tar.shape[2]!=raw.shape[2]:
                tar = trnf.interpolate(tar, [raw.shape[2], raw.shape[3]], mode="bilinear");
        return this.speed*neko_masked_l2(raw, tar);

        # l2 needs to normalize it a bit
#
# import cv2
# import numpy as np
# from neko_sdk.encoders.dcgan import DCGAN_reconstruction_network,DCGAN_reconstruction_network_insnorm
# from neko_sdk.loss_functions.balanced_l2 import neko_masked_l2,neko_masked_l2_nr
# # balanced tanh l2 loss.
# from torch import nn;
# import torch;
# from torch.nn import functional as trnf
# from neko_sdk.neko_score_merging import scatter_cvt;
#
#
#
#
# class neko_dcgan_reconstruction(nn.Module):
#     def __init__(this,psize,gsize=64,expf=1,och=1):
#         super(neko_dcgan_reconstruction,this).__init__();
#         this.recon=DCGAN_reconstruction_network(psize,gsize,och*expf*expf);
#         if(expf>1):
#             this.reconpost=torch.nn.PixelShuffle(expf);
#         else:
#             this.reconpost=None;
#         pass;
#
#     def forward(this,protos):
#         raw =  this.recon(protos);
#         if(this.reconpost is not None):
#             raw=this.reconpost(raw);
#         return raw;
#
# class neko_dcgan_reconstructionN(nn.Module):
#     def __init__(this, psize, gsize=64, expf=1, och=1,scale=1.5,bias=0):
#         super(neko_dcgan_reconstructionN, this).__init__();
#         this.scale=scale;
#         this.bias=bias;
#
#         this.recon = DCGAN_reconstruction_network(psize, gsize, och * expf * expf);
#         if (expf > 1):
#             this.reconpost = torch.nn.PixelShuffle(expf);
#         else:
#             this.reconpost = None;
#         pass;
#
#     def forward(this, protos):
#         raw = this.recon(protos);
#         if (this.reconpost is not None):
#             raw = this.reconpost(raw);
#         return trnf.tanh(raw)*this.scale+this.bias;
#
# class neko_dcgan_reconstructionN_insnorm(nn.Module):
#     def __init__(this, psize, gsize=64, expf=1, och=1, scale=1.5, bias=0):
#         super(neko_dcgan_reconstructionN_insnorm, this).__init__();
#         this.scale = scale;
#         this.bias = bias;
#
#         this.recon = DCGAN_reconstruction_network_insnorm(psize, gsize, och * expf * expf);
#         if (expf > 1):
#             this.reconpost = torch.nn.PixelShuffle(expf);
#         else:
#             this.reconpost = None;
#         pass;
#
#     def forward(this, protos):
#         raw = this.recon(protos);
#         if (this.reconpost is not None):
#             raw = this.reconpost(raw);
#         return trnf.tanh(raw) * this.scale + this.bias;
#
#         # l2 needs to normalize it a bit
# class neko_masked_l2_loss(nn.Module):
#     def __init__(this,speed=1):
#         super(neko_masked_l2_loss,this).__init__();
#         this.speed=speed;
#
#     def debug(this,raw,dbgkey):
#         raww=(raw[10:20].reshape(640,64).detach()*127.5+128).clip(0,255).detach().cpu().numpy().astype(np.uint8);
#         cv2.namedWindow("preview"+str(dbgkey));
#         cv2.imshow("preview"+str(dbgkey),raww);
#
#         cv2.waitKey(0);
#
#     def forward(this,raw,gt,dbgkey=None):
#         if(dbgkey):
#             this.debug(raw,dbgkey);
#         with torch.no_grad():
#             tar=gt.to(raw.device);
#             if tar.shape[2]!=raw.shape[2]:
#                 tar = trnf.interpolate(tar, [raw.shape[2], raw.shape[3]], mode="bilinear");
#         return this.speed*neko_masked_l2(raw, tar);
#
# class neko_masked_l2_loss_matched(nn.Module):
#     def __init__(this,speed=1):
#         super(neko_masked_l2_loss_matched,this).__init__();
#         this.speed=speed;
#
#     def debug(this,raw,dbgkey):
#         raww=(raw[10:20].reshape(640,64).detach()*127.5+128).clip(0,255).detach().cpu().numpy().astype(np.uint8);
#         cv2.namedWindow("preview"+str(dbgkey));
#         cv2.imshow("preview"+str(dbgkey),raww);
#
#         cv2.waitKey(0);
#
#     def forward(this,raw,gt,plabel,dbgkey=None):
#         if(dbgkey):
#             this.debug(raw,dbgkey);
#         with torch.no_grad():
#             tar=gt.to(raw.device);
#             if tar.shape[2]!=raw.shape[2]:
#                 tar = trnf.interpolate(tar, [raw.shape[2], raw.shape[3]], mode="bilinear");
#         return this.speed*neko_masked_l2(raw, tar);
#
#
#
#
# class neko_masked_l2_loss_nr(nn.Module):
#     def __init__(this,speed=1):
#         super(neko_masked_l2_loss_nr,this).__init__();
#         this.speed=speed;
#
#     def debug(this,raw,dbgkey):
#         raww=(raw[10:20].reshape(640,64).detach()*127.5+128).clip(0,255).detach().cpu().numpy().astype(np.uint8);
#         cv2.namedWindow("preview"+str(dbgkey));
#         cv2.imshow("preview"+str(dbgkey),raww);
#
#         cv2.waitKey(0);
#
#     def forward(this,raw,gt,dbgkey=None):
#         if(dbgkey):
#             this.debug(raw,dbgkey);
#         with torch.no_grad():
#             tar=gt.to(raw.device);
#             if tar.shape[2]!=raw.shape[2]:
#                 tar = trnf.interpolate(tar, [raw.shape[2], raw.shape[3]], mode="bilinear");
#         return this.speed*neko_masked_l2_nr(raw, tar);
# class neko_masked_l2_loss_reduction2(nn.Module):
#     def __init__(this,speed=1):
#         super(neko_masked_l2_loss_reduction2,this).__init__();
#         this.speed=speed;
#
#     def debug(this,raw,dbgkey):
#         raww=(raw[10:20].reshape(640,64).detach()*127.5+128).clip(0,255).detach().cpu().numpy().astype(np.uint8);
#         cv2.namedWindow("preview"+str(dbgkey));
#         cv2.imshow("preview"+str(dbgkey),raww);
#
#         cv2.waitKey(0);
#
#     def forward(this,raw,gt,gtlabel,label,dbgkey=None):
#         if(dbgkey):
#             this.debug(raw,dbgkey);
#         with torch.no_grad():
#             tar=gt.to(raw.device);
#             if tar.shape[2]!=raw.shape[2]:
#                 tar = trnf.interpolate(tar, [raw.shape[2], raw.shape[3]], mode="bilinear");
#         # because the scatter, we cannot take gt, as we don't known which gt to take.
#         mscore=-neko_masked_l2_nr(raw.reshape(1,raw.shape[0],raw.shape[2],raw.shape[3]), tar);
#         merged_scr=-scatter_cvt(mscore,gtlabel,dim=0)
#         merged=merged_scr.take(label);
#         return this.speed*merged;
#
# class neko_masked_l2_loss_reduction(nn.Module):
#     def __init__(this,speed=1):
#         super(neko_masked_l2_loss_reduction,this).__init__();
#         this.speed=speed;
#
#     def debug(this,raw,dbgkey):
#         raww=(raw[10:20].reshape(640,64).detach()*127.5+128).clip(0,255).detach().cpu().numpy().astype(np.uint8);
#         cv2.namedWindow("preview"+str(dbgkey));
#         cv2.imshow("preview"+str(dbgkey),raww);
#
#         cv2.waitKey(0);
#
#     def forward(this,raw,gt,gtlabel,label,dbgkey=None):
#         if(dbgkey):
#             this.debug(raw,dbgkey);
#         with torch.no_grad():
#             tar=gt.to(raw.device);
#             if tar.shape[2]!=raw.shape[2]:
#                 tar = trnf.interpolate(tar, [raw.shape[2], raw.shape[3]], mode="bilinear");
#         # because the scatter, we cannot take gt, as we don't known which gt to take.
#         mscore=-neko_masked_l2_nr(raw.reshape(1,raw.shape[0],raw.shape[2],raw.shape[3]), tar);
#         merged_scr=-scatter_cvt(mscore,gtlabel,dim=0)
#         merged=merged_scr.take(label);
#         return this.speed*merged;
#
# class neko_btl2_reconstruction_loss(nn.Module):
#     def __init__(this,psize,gsize=64,expf=1):
#         super(neko_btl2_reconstruction_loss,this).__init__();
#         this.recon=neko_dcgan_reconstruction(psize,gsize,expf);
#     def masked_l2(this,raw,vprotos):
#         return neko_masked_l2(raw,vprotos);
#     def debug(this,raw,dbgkey):
#         raww=(raw[10:20].reshape(640,64).detach()*127.5+128).clip(0,255).detach().cpu().numpy().astype(np.uint8);
#         cv2.namedWindow("preview"+str(dbgkey));
#         cv2.imshow("preview"+str(dbgkey),raww);
#
#         cv2.waitKey(0);
#
#
#     def forward(this,protos,vprotos=None,dbgkey=None):
#         raw =  this.recon(protos);
#         if(dbgkey):
#             this.debug(raw,dbgkey);
#         with torch.no_grad():
#             tar=vprotos.to(raw.device);
#             if tar.shape[2]!=raw.shape[2]:
#                 tar=trnf.interpolate(tar,[raw.shape[2],raw.shape[3]],mode="bilinear");
#         return this.masked_l2(raw,tar);
#
#         # l2 needs to normalize it a bit
