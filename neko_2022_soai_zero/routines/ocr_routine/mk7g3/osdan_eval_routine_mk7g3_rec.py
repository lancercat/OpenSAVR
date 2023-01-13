import random

import torch
from torch.nn import functional as trnf;
import numpy as np;
from neko_2021_mjt.routines.ocr_routines.mk7.osdan_routine_mk7 import neko_HDOS2C_eval_routine_CFmk7
from neko_2021_mjt.routines.subroutines.fe_seq.GTA import dump_att_ims
import regex

class neko_HDOS2C_eval_routine_CFmk7g3_rec(neko_HDOS2C_eval_routine_CFmk7):
    def recon2img(this,out_emb,modular_dict,proto,pred_length,recon_name="p_recon"):
        preds=modular_dict["preds"];
        flogits = preds[0](out_emb, proto, None);
        flogits, _ = this.inflater.inflate(flogits, pred_length);
        seq_emb, segs = this.inflater.inflate(out_emb, pred_length);
        pids = flogits[:, :-1].argmax(dim=1);
        seq_pro = proto[pids]
        seq_emb = trnf.normalize(seq_emb, dim=1);
        seq_pro_rec=modular_dict[recon_name].model.recon(seq_pro.unsqueeze(-1).unsqueeze(-1));
        seq_emb_rec=modular_dict[recon_name].model.recon(seq_emb.unsqueeze(-1).unsqueeze(-1))
        seq_pro_rec = trnf.sigmoid(seq_pro_rec);
        seq_emb_rec = trnf.sigmoid(seq_emb_rec);
        seqrec=torch.cat([seq_emb_rec,seq_pro_rec],2)*255;
        seqrec=seqrec.split(list(segs.cpu().numpy()));
        seqrec=[s.permute(2,0,3,1).reshape(s.shape[2],s.shape[3]*s.shape[0],s.shape[1]).detach().cpu().numpy().astype(np.uint8) for s in seqrec];
        return seqrec;


    def test_impl(this,data_dict, modular_dict,logger_dict):
        data,label,proto, plabel, tdict= \
        data_dict["image"],data_dict["label"],data_dict["proto"],data_dict["plabel"],data_dict["tdict"];
        preds=modular_dict["preds"];
        seq=modular_dict["seq"];
        sampler=modular_dict["sampler"];
        data=data.cuda();
        features = modular_dict["feature_extractor"](data)
        A,pred_length = modular_dict["CAM"](features)
        pred_length=pred_length.argmax(dim=-1)
        # A0=A.detach().clone();
        out_emb =seq(features[-1],A,None);
        # lossess = []
        beams_ = [];
        probs = [];
        # terms = [];
        if(modular_dict["p_recon"]!="NEP_skipped_NEP"):
            recon_name = "p_recon"
        else:
            recon_name = "shuf_recon"
        try:
            prec=this.recon2img(out_emb,modular_dict,proto,pred_length,recon_name);
        except:
            prec=None;

        loss = 0;
        for i in range(len(preds)):
            logits = preds[i](out_emb, proto,plabel);
            logits,_ = this.inflater.inflate(logits, pred_length);

            choutput, prdt_prob = sampler.model.decode(logits, pred_length, proto, plabel, tdict);
            beams_.append(choutput);
            probs.append(prdt_prob);
            # loss_, terms_ = modular_dict["losses"][i](proto, preds, label_flatten);
            # loss = loss_ + loss;
            beams_.append(choutput);
            # terms.append(terms_);
        beams=[];
        for i in range(features[-1].shape[0]):
            beam = [];
            for j in range(len(beams_)):
                beam.append(beams_[j][i]);
            beams.append(beam)
        # A=A.max(dim=2)[0];
        flabel=[];
        if(label is not None):
            for l in label:
                s="";
                for c in regex.findall(r'\X', l, regex.U) :
                    if(c not in tdict):
                        s+="⑨";
                    else:
                        s+=c;
                flabel.append(s);
            logger_dict["accr"].add_iter(beams_[0],pred_length, flabel)
        try:
            aim = dump_att_ims(data, None, A, pred_length, label);
            all = [{"att": a, "prec": p} for a,p in zip(aim,prec)];
            seq_emb, pids = this.dump_pred_raw(out_emb, modular_dict, proto, pred_length);
            all_feat = [{"feat_res": (f,l)} for f,l in zip(seq_emb,pids)];
            rdict={"xtra_ims": all,"xtra_pts":all_feat};
            rdict["label_override"]=flabel;
        except:
            # override unknown with unknown.
            rdict={}
            rdict["label_override"]=flabel;

        return beams_[0], rdict, beams;


    def test_impl(this,data_dict, modular_dict,logger_dict):
        data,label,proto, plabel, tdict= \
        data_dict["image"],data_dict["label"],data_dict["proto"],data_dict["plabel"],data_dict["tdict"];
        preds=modular_dict["preds"];
        seq=modular_dict["seq"];
        sampler=modular_dict["sampler"];
        data=data.cuda();
        features = modular_dict["feature_extractor"](data)
        A,pred_length = modular_dict["CAM"](features)
        pred_length=pred_length.argmax(dim=-1)
        # A0=A.detach().clone();
        out_emb =seq(features[-1],A,None);
        # lossess = []
        beams_ = [];
        probs = [];
        # terms = [];
        if(modular_dict["p_recon"]!="NEP_skipped_NEP"):
            recon_name = "p_recon"
        else:
            recon_name = "shuf_recon"
        try:
            prec=this.recon2img(out_emb,modular_dict,proto,pred_length,recon_name);
        except:
            prec=None;

        loss = 0;
        for i in range(len(preds)):
            logits = preds[i](out_emb, proto,plabel);
            logits,_ = this.inflater.inflate(logits, pred_length);

            choutput, prdt_prob = sampler.model.decode(logits, pred_length, proto, plabel, tdict);
            beams_.append(choutput);
            probs.append(prdt_prob);
            # loss_, terms_ = modular_dict["losses"][i](proto, preds, label_flatten);
            # loss = loss_ + loss;
            beams_.append(choutput);
            # terms.append(terms_);
        beams=[];
        for i in range(features[-1].shape[0]):
            beam = [];
            for j in range(len(beams_)):
                beam.append(beams_[j][i]);
            beams.append(beam)
        # A=A.max(dim=2)[0];
        flabel=[];
        if(label is not None):
            for l in label:
                s="";
                for c in regex.findall(r'\X', l, regex.U) :
                    if(c not in tdict):
                        s+="⑨";
                    else:
                        s+=c;
                flabel.append(s);
            logger_dict["accr"].add_iter(beams_[0],pred_length, flabel)
        try:
            aim = dump_att_ims(data, None, A, pred_length, label);
            all = [{"att": a, "prec": p} for a,p in zip(aim,prec)];
            seq_emb, pids = this.dump_pred_raw(out_emb, modular_dict, proto, pred_length);
            all_feat = [{"feat_res": (f,l)} for f,l in zip(seq_emb,pids)];
            rdict={"xtra_ims": all,"xtra_pts":all_feat};
            rdict["label_override"]=flabel;
        except:
            # override unknown with unknown.
            rdict={}
            rdict["label_override"]=flabel;

        return beams_[0], rdict, beams;


class neko_HDOS2C_eval_routine_CFmk7g3_rec_anc(neko_HDOS2C_eval_routine_CFmk7g3_rec):


    def deform2img(this,img,beacon,deform,vissz=[128,256]):
        N,_,H,W=deform.shape;
        sx,sy,theta=deform.split([1,1,1],dim=1);
        sx=trnf.interpolate(sx,[img.shape[2],img.shape[3]]);
        sy=trnf.interpolate(sy, [img.shape[2], img.shape[3]]);
        theta=trnf.interpolate(theta, [img.shape[2], img.shape[3]]);
        cim=img.cpu();
        isx=torch.cat([cim*255,(sx.detach()*80).repeat(1,3,1,1)],dim=2).permute(0,2,3,1).cpu().numpy().astype(np.uint8);
        isy=torch.cat([cim*255,(sy.detach()*80).repeat(1,3,1,1)],dim=2).permute(0,2,3,1).cpu().numpy().astype(np.uint8);

        # cos=torch.cos(theta);
        # sin=torch.sin(theta);
        # w00 = sx * cos;
        # w01 = -sy * sin;
        # w10 = sx * sin;
        # # it's an delta.
        # w11 = sy * cos - 1;
        #
        # mat=torch.zeros(N*H*W,3,3,device=w00.device);
        # mat[:,0,0]=w00.reshape(-1);
        # mat[:, 0, 1] = w01.reshape(-1);
        # mat[:,1,0]=w10.reshape(-1);
        # mat[:, 1, 1] = w11.reshape(-1);


        return isx,isy;

    def test_impl(this,data_dict, modular_dict,logger_dict):
        data,beacon,label,proto, plabel, tdict= \
        data_dict["image"],data_dict["beacon"],data_dict["label"],data_dict["proto"],data_dict["plabel"],data_dict["tdict"];
        preds=modular_dict["preds"];
        seq=modular_dict["seq"];
        sampler=modular_dict["sampler"];
        data,beacon=data.cuda(),beacon.cuda();
        features,deform = modular_dict["feature_extractor"](data,beacon,True)
        A,pred_length = modular_dict["CAM"](features)
        pred_length=pred_length.argmax(dim=-1)
        # A0=A.detach().clone();
        out_emb =seq(features[-1],A,None);
        # lossess = []
        beams_ = [];
        probs = [];
        # terms = [];
        if(modular_dict["p_recon"]=="NEP_skipped_NEP"):
            prec=None;
        else:
            prec=this.recon2img(out_emb,modular_dict,proto,pred_length);
        dvimx,dvimy=this.deform2img(data,beacon,deform);

        loss = 0;
        for i in range(len(preds)):
            logits = preds[i](out_emb, proto,plabel);
            logits,_ = this.inflater.inflate(logits, pred_length);

            choutput, prdt_prob = sampler.model.decode(logits, pred_length, proto, plabel, tdict);
            beams_.append(choutput);
            probs.append(prdt_prob);
            # loss_, terms_ = modular_dict["losses"][i](proto, preds, label_flatten);
            # loss = loss_ + loss;
            beams_.append(choutput);
            # terms.append(terms_);
        beams=[];
        for i in range(features[-1].shape[0]):
            beam = [];
            for j in range(len(beams_)):
                beam.append(beams_[j][i]);
            beams.append(beam)
        # A=A.max(dim=2)[0];
        flabel=[];
        if(label is not None):
            for l in label:
                s="";
                for c in regex.findall(r'\X', l, regex.U) :
                    if(c not in tdict):
                        s+="⑨";
                    else:
                        s+=c;
                flabel.append(s);
            logger_dict["accr"].add_iter(beams_[0],pred_length, flabel)
        try:
            aim = dump_att_ims(data, None, A, pred_length, label);
            if(prec is not None):
                all_im = [{"att": a,"scale_x": sx, "scale_y": sy, "prec": p} for a,sx,sy,p in zip(aim,dvimx,dvimy,prec)];
            else:
                all_im = [{"att": a ,"scale_x": sx, "scale_y": sy} for a,sx,sy in zip(aim,dvimx,dvimy)];
            seq_emb, pids = this.dump_pred_raw(out_emb, modular_dict, proto, pred_length);
            all_feat = [{"feat_res": (f,l)} for f,l in zip(seq_emb,pids)];
            rdict={"xtra_ims": all_im,"xtra_pts":all_feat};
        except:
            rdict={}
        return beams_[0], rdict, beams;

