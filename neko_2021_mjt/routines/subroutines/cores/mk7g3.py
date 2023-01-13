
# using mk5 DTD
import torch
import torch.nn.functional as trnf
from neko_2020nocr.dan.common.common import flatten_label
from neko_sdk.ocr_modules.io.encdec import decode_prob
from neko_2021_mjt.modulars.neko_inflater import neko_inflater


def recog_loss(modular_dict,pred_length,culength,fout_emb,proto,plabel,label_flatten,length,tdict):
    preds = modular_dict["preds"];
    loss = trnf.cross_entropy(pred_length, culength);
    logits_list = [];
    terms = {};
    beams = [];
    for i in range(len(preds)):
        logits = preds[i](fout_emb, proto, plabel);
        loss_, terms_ = modular_dict["losses"][i](proto, logits, label_flatten);
        loss = loss_ + loss;
        terms[str(i)]=terms_;
        logits_list.append(logits.detach());
        choutput, prdt_prob = decode_prob(logits, length, tdict);
        beams.append(choutput);
    return loss,terms,beams;


class neko_HDOS2C_routine_CFmk7g3_core:
    def fe_seq(this,clips,modular_dict,length):
        features = modular_dict["feature_extractor"](clips)
        features=[f.contiguous() for f in features];
        A,pred_length = modular_dict["CAM"](features)
        out_emb = modular_dict["seq"](features[-1], A, length);
        return out_emb,A,pred_length;
    def arm_submodules(this):
        this.inflater = neko_inflater();

    def __init__(this):
        this.arm_submodules();

    def fp_impl(this, input_dict,exdict, modular_dict,logger_dict,device):
        clips=input_dict["image"];

        # Prototypes(sampled)
        # And this helps using SYNTH words in LSCT
        target=exdict["target"];
        length=exdict["length"];
        tdict=exdict["tdict"];
        normprotos=exdict["proto"];
        # semb=exdict["semb"];
        plabel=exdict["plabel"];

        prototyper=modular_dict["prototyper"]

        proto=prototyper(normprotos,use_sp=False);
        label_flatten, length = flatten_label(target,EOSlen=0,length=length);
        target, label_flatten,culength = target.to(device), label_flatten.to(device),length.long().to(device);
        out_emb,A,pred_length=this.fe_seq(clips.to(device),modular_dict,length);
        fout_emb,_=this.inflater.inflate(out_emb,length)

        loss,terms,beams=recog_loss(modular_dict,pred_length,culength,fout_emb,proto,plabel,label_flatten,length,tdict);

        # computing accuracy and loss
        tarswunk = ["".join([tdict[i.item()] for i in target[j]]).replace('[s]', "") for j in
                    range(len(target))];
        logger_dict["accr"].add_iter(beams[0], length, tarswunk)
        logger_dict["loss"].add_iter(loss, terms[0])
        return loss;