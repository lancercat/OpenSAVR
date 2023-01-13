import torch
import cv2
from torch.nn import functional as trnf
import numpy as np
from neko_sdk.ocr_modules.result_renderer import render_word
def dump_att_im(clip,GA,TA,length,gt,tdict=None,pr=None,scharset=None):
    img = (clip.permute(1, 2, 0) * 255).detach().cpu().numpy();
    if(pr is not None):
        if(scharset is None):
            scharset=tdict;
        red, ned = render_word(tdict, scharset, img, gt.lower(), pr.lower());
    else:
        red=None;
    TA = trnf.interpolate(TA.unsqueeze(0), [32, 128]).squeeze(0).cpu();
    TIs = [clip.cpu()];
    if(GA is not None):
        GA = trnf.interpolate(GA.unsqueeze(0), [32, 128]).squeeze(0);
        GI = (GA.cpu() * 0.9 + 0.1) * clip.cpu();
        TIs.append(GI);

    for j in range(length):
        TIs.append(clip.cpu() * (TA[ j:j + 1] * 0.9 + 0.1));
    TIs.append(torch.max(TA[:length+1],dim=0)[0].repeat([3,1,1]));
    if(red is None):
        return(torch.cat(TIs, 1).permute(1, 2, 0) * 255).detach().cpu().numpy().astype(np.uint8);
    else:
        dim=(torch.cat(TIs, 1).permute(1, 2, 0) * 255).detach().cpu().numpy().astype(np.uint8);
        dh=int(dim.shape[0]*(red.shape[1]/dim.shape[1]));
        return np.concatenate([red,cv2.resize(dim,(red.shape[1],dh))]);
def dump_mask_im(clip,mask):
    ret = torch.cat([
        (clip.permute(1, 2, 0) * 255).detach(),
        (mask.repeat(3,1,1).permute(1, 2, 0) * 255).detach()],0);
    return ret.cpu().numpy();



def dump_att_ims(clips,GA,TA_,length_,gt,tdict=None,pr=None,scharset=None):
    tis=[];
    if(len(TA_.shape)==5):
        length=length_*TA_.shape[2];
        TA=TA_.reshape(TA_.shape[0],TA_.shape[1]*TA_.shape[2],TA_.shape[3],TA_.shape[4]);
    else:
        length=length_;
        TA=TA_;
    for i in range(clips.shape[0]):
        if(GA is None):
            aga=None;
        else:
            aga=GA[i];
        if(gt is None):
            agt=None;
        else:
            agt=gt[i]
        tis.append(dump_att_im(clips[i], aga, TA[i], length[i], agt));
    return tis;
def dump_mask_ims(clips,maskes):
    tis=[];
    for i in range(clips.shape[0]):
        tis.append(dump_mask_im(clips[i],maskes[i]));
    return tis;

def debug_gta(clips,GA,TA,p_len,dbgkey):
    for i in range(clips.shape[0]):
        ti = dump_att_im(clips[i], GA[i], TA[i], p_len[i], None);
        cv2.namedWindow("a"+dbgkey, 0);
        cv2.imshow("a"+dbgkey, ti);
        cv2.waitKey(10);
# used in mk7 routines and later.
# this change allows va8 to provide a pixel level control over image splitting, hopefully can reduce some variance
