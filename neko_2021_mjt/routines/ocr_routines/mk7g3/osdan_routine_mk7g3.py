
# using mk5 DTD
from neko_sdk.MJT.neko_abstract_routines import neko_abstract_routine;
from neko_2020nocr.dan.utils import Loss_counter,neko_os_Attention_AR_counter
import numpy as np
import cv2;
from neko_sdk.ocr_modules.io.encdec import decode_prob
from neko_2021_mjt.routines.subroutines.cores.mk7g3 import neko_HDOS2C_routine_CFmk7g3_core


# mk5 CF branch dropped predict-sample-predict support.
# A GP branch will be added if it's ever to be supported
# Mk7 CF branch uses CAM to perform length prediction, [s] is no more needed
# Still training only.
# It does not try to make prototype, which makes the lsct more idiot-proof
# G3 cores are for quick recasting
class neko_HDOS2C_routine_CFmk7g3(neko_abstract_routine):
    def mk_proto(this,label,sampler):
        normprotos, plabel, tdict=sampler.model.sample_charset_by_text(label,use_sp=False)
        semb=None
        return normprotos, semb, plabel, tdict

    def set_etc(this, args):
        this.maxT = args["maxT"];
        this.core=neko_HDOS2C_routine_CFmk7g3_core();

    def set_loggers(this, log_path, log_each, name):
        this.logger_dict = {
            "accr": neko_os_Attention_AR_counter("[" + name + "]" + "train_accr", False),
            "loss": Loss_counter("[" + name + "]" + "train_accr"),
        };

    def show_clip(this, clip, label):
        im = (clip.detach() * 255).permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8);
        for i in range(len(im)):
            cv2.imshow(label[i], im[i]);
        cv2.waitKey(0);
    def fp_impl(this, input_dict, modular_dict, logger_dict, nEpoch, batch_idx, device):
        sampler=modular_dict["sampler"];
        normprotos, semb, plabel, tdict=this.mk_proto(input_dict["label"],sampler);
        target, length = sampler.model.encode_noeos(normprotos, plabel, tdict, input_dict["label"]);
        exdict={};
        exdict["proto"]=normprotos;
        exdict["target"]=target;
        exdict["length"]=length;
        exdict["plabel"]=plabel;
        exdict["tdict"]=tdict;
        exdict["semb"] = semb;
        loss=this.core.fp_impl(input_dict,exdict,modular_dict,logger_dict,device)
        return loss;

class neko_HDOS2C_routine_CFmk7g3_rec(neko_HDOS2C_routine_CFmk7g3):
    def set_etc(this, args):
        this.maxT = args["maxT"];
        this.core=neko_HDOS2C_routine_CFmk7g3_core();
