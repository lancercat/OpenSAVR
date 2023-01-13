
# using mk5 DTD
import torch
import torch.nn.functional as trnf
import regex;
from neko_sdk.MJT.neko_abstract_routines import neko_abstract_eval_routine;
from neko_2020nocr.dan.utils import Loss_counter,neko_os_Attention_AR_counter,neko_oswr_Attention_AR_counter
import numpy as np
import cv2;
from neko_2021_mjt.modulars.neko_inflater import neko_inflater
from neko_2021_mjt.routines.subroutines.fe_seq.GTA import dump_att_ims


class neko_HDOS2C_eval_routine_CFmk7(neko_abstract_eval_routine):
    def dump_pred_raw(this,out_emb,modular_dict,proto,pred_length):
        nT, nB = out_emb.shape[0], out_emb.shape[1];
        preds=modular_dict["preds"];
        flogits = preds[0](out_emb.reshape([nT * nB] +list(out_emb.shape[2:])), proto, None).reshape([nT, nB, -1]);
        flogits, _ = this.inflater.inflate(flogits, pred_length);
        seq_emb, segs = this.inflater.inflate(out_emb, pred_length);
        segs=list(segs.cpu().detach().numpy());
        pids = flogits[:, :-1].argmax(dim=1).detach().cpu();
        seq_emb=seq_emb.detach().cpu();
        pids=torch.split(pids,segs);
        seq_emb=torch.split(seq_emb,segs);
        return seq_emb,pids

    def set_etc(this, args):
        this.maxT = args["maxT"];
        this.inflater=neko_inflater();

    def set_loggers(this, log_path, name, args):
        try:
            if (args["measure_rej"]==True):
                this.logger_dict = {"accr":neko_oswr_Attention_AR_counter("[" + name + "]" + "test_accr", False),
                                    }
            else:
                this.logger_dict = {
                    "accr": neko_os_Attention_AR_counter("[" + name + "]" + "test_accr", False),
                    "loss": Loss_counter("[" + name + "]" + "train_accr"),
                };
        except:
            this.logger_dict={
                "accr": neko_os_Attention_AR_counter("[" + name + "]" + "test_accr", False),
                "loss": Loss_counter("[" + name + "]" + "train_accr"),
            };
    def test_topk_impl(this, data_dict, modular_dict, logger_dict,k):

        data, label, proto, plabel, tdict = \
            data_dict["image"], data_dict["label"], data_dict["proto"], data_dict["plabel"], data_dict["tdict"];
        preds = modular_dict["preds"];
        seq = modular_dict["seq"];
        sampler = modular_dict["sampler"];

        data = data.cuda();
        features = modular_dict["feature_extractor"](data)
        A, pred_length = modular_dict["CAM"](features)
        pred_length = pred_length.argmax(dim=-1)
        # A0=A.detach().clone();
        out_emb = seq(features[-1], A, None);
        # lossess = []
        beams_ = [];
        probs = [];
        # terms = [];

        loss = 0;
        nT, nB = out_emb.shape[0], out_emb.shape[1];
        logits = preds[0](out_emb.reshape([nT * nB, -1]), proto, plabel).reshape([nT, nB, -1]);
        logits, _ = this.inflater.inflate(logits, pred_length);
        idmat,beams = sampler.model.decode_beam_char(logits, pred_length, proto, plabel, tdict);
            # terms.append(terms_);

        return  idmat,beams,A;

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
        seq_emb,pids=this.dump_pred_raw(out_emb,modular_dict,proto,pred_length);

        loss = 0;
        for i in range(len(preds)):
            nT,nB=out_emb.shape[0],out_emb.shape[1];
            logits = preds[i](out_emb.reshape([nT*nB]+list(out_emb.shape[2:])), proto,plabel).reshape([nT,nB,-1]);
            logits,_ = this.inflater.inflate(logits, pred_length);
            choutput, prdt_prob = sampler.model.decode(logits, pred_length, proto, plabel, tdict);
            beams_.append(choutput);
            probs.append(prdt_prob);
            # loss_, terms_ = modular_dict["losses"][i](proto, preds, label_flatten);
            # loss = loss_ +  loss;
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
            all_im = [{"att": im} for im in aim];
            all_feat = [{"feat_res": (f,l)} for f,l in zip(seq_emb,pids)];
            rdict={"xtra_ims": all_im,"xtra_pts":all_feat};
        except:
            rdict={}
        return beams_[0], rdict, beams;

        # A.detach().reshape(A.shape[0], A.shape[1], A.shape[2], A.shape[3]).sum(2)
    def vis_logits_impl(this,img,data_dict,modular_dict,at_time):

        _, label, proto, plabel, tdict = \
            data_dict["image"], data_dict["label"], data_dict["proto"], data_dict["plabel"], data_dict["tdict"];
        data=img
        preds=modular_dict["preds"];
        seq=modular_dict["seq"];
        sampler=modular_dict["sampler"];


        data=data.cuda();
        data=torch.nn.Parameter(data,requires_grad=True);

        features = modular_dict["feature_extractor"](data)
        A,pred_length = modular_dict["CAM"](features)
        pred_length=pred_length.argmax(dim=-1)
        # A0=A.detach().clone();
        out_emb =seq(features[-1],A,None);
        # lossess = []
        beams_ = [];
        probs = [];
        # terms = [];

        loss = 0;
        nT,nB=out_emb.shape[0],out_emb.shape[1];
        logits = preds[0](out_emb.reshape([nT*nB]+list(out_emb.shape[2:])), proto,plabel).reshape([nT,nB,-1]);
        logits, _ = this.inflater.inflate(logits, pred_length);

        if (len(logits) <= at_time):
            return None;
        return logits[at_time]

    def pretest_impl(this,modular_dict,metaargs,**kwargs):
        rot = kwargs["rot"];
        normproto, plabel, tdict = modular_dict["sampler"].model.dump_all(metaargs=metaargs,use_sp=False);
        if (not rot):
            proto = modular_dict["prototyper"](normproto,use_sp=False);
        else:
            proto = modular_dict["prototyper"](normproto, rot);
        return {"proto":proto,"plabel":plabel,"tdict":tdict};



