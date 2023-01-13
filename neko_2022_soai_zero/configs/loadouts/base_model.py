from neko_2021_mjt.configs.bogo_modules.config_res_binorm import config_bogo_resbinorm;
from neko_2021_mjt.configs.common_subs.arm_post_fe_shared_prototyper import arm_shared_prototyper_np
from neko_2021_mjt.configs.modules.config_sa import config_sa_mk3
from  neko_2021_mjt.configs.modules.config_fe_db import config_fe_r45_binorm_orig,config_fe_r45_binorm_ptpt;
from neko_2021_mjt.configs.modules.config_cam_stop import config_cam_stop
from neko_2021_mjt.configs.modules.config_ocr_sampler import config_ocr_sampler
from neko_2021_mjt.configs.modules.config_dtd_xos_mk5 import config_dtdmk5
from neko_2021_mjt.configs.modules.config_ospred import config_linxos
from neko_2021_mjt.configs.modules.config_cls_emb_loss import config_cls_emb_loss2

def arm_common_part_noloss(srcdst,prefix,maxT,capacity,feat_ch,tr_meta_path,expf=1,TA_scales=None,detached_ta=False,detached_ga_proto=False,dropp=False):
    if (TA_scales == None):
        TA_scales = [
            [int(32*expf), 16, 64],
            [int(128*expf), 8, 32],
            [int(feat_ch*expf), 8, 32]
        ]
    srcdst[prefix + "GA"] = config_sa_mk3(feat_ch=TA_scales[0][0]);

    srcdst[prefix + "TA"] = config_cam_stop(maxT, feat_ch=feat_ch, scales=TA_scales, detached=detached_ta);
    srcdst[prefix + "Latin_62_sampler"] = config_ocr_sampler(tr_meta_path, capacity);
    srcdst[prefix + "DTD"] = config_dtdmk5();
    srcdst[prefix + "pred"] = config_linxos();
    srcdst = arm_shared_prototyper_np(
        srcdst, prefix, capacity, feat_ch,
        prefix + "feature_extractor_proto",
        prefix + "GA",
        use_sp=False,
        detached_ga=detached_ga_proto,
        drop=dropp
    );
    # look, explicitly skipping means that you are aware what might be skipped,
    # if a routine try to utilize a module you did not instruct to skip,
    # the program crashes to prevent unexpected behaviours.
    srcdst[prefix + "ctxmod"] = "NEP_skipped_NEP";
    srcdst[prefix + "dom_mix"] = "NEP_skipped_NEP";
    srcdst[prefix + "p_recon"] = "NEP_skipped_NEP";
    srcdst[prefix + "p_recon_loss"] = "NEP_skipped_NEP";
    srcdst[prefix + "f_recon"] = "NEP_skipped_NEP";
    srcdst[prefix + "f_recon_loss"] = "NEP_skipped_NEP";
    srcdst[prefix + "fpm_recon_loss"] = "NEP_skipped_NEP";
    srcdst[prefix + "shuf_img"] = "NEP_skipped_NEP";
    srcdst[prefix + "recon_char_fe"]= "NEP_skipped_NEP";
    srcdst[prefix + "recon_char_pred"]= "NEP_skipped_NEP";
    srcdst[prefix + "recon_mva"] = "NEP_skipped_NEP";
    srcdst[prefix+ "shuf_recon"] = "NEP_skipped_NEP";
    srcdst[prefix+ "shuf_recon_loss"] = "NEP_skipped_NEP";
    srcdst[prefix+"shuf_sim_loss"]="NEP_skipped_NEP";
    srcdst[prefix + "shuf_part_recon"] = "NEP_skipped_NEP";
    srcdst[prefix+"proto_part_recon"]="NEP_skipped_NEP";
    srcdst[prefix + "proto_part_recon_loss"] = "NEP_skipped_NEP";
    srcdst[prefix + "shuf_part_recon_loss"] = "NEP_skipped_NEP";
    srcdst[prefix+"shuf_proto"]= "NEP_skipped_NEP";
    srcdst[prefix+"inv_loss"]="NEP_skipped_NEP"
    return srcdst;


def arm_common_part(srcdst,prefix,maxT,capacity,feat_ch,tr_meta_path,expf=1,fecnt=3,wemb=0.3,wrej=0.1,inplace=True,TA_scales=None,detached_ta=False,detached_ga_proto=False,dropp=False):
    srcdst=arm_common_part_noloss(srcdst,prefix,maxT,capacity,feat_ch,tr_meta_path,expf=expf,TA_scales=TA_scales,detached_ta=detached_ta,detached_ga_proto=detached_ga_proto,dropp=dropp)
    srcdst[prefix + "loss_cls_emb"] = config_cls_emb_loss2(wemb, wrej);
    return srcdst;



def arm_base_fe_module(srcdst,feat_ch,fecnt,prefix,inplace,drop=0,expf=1):
    if(type(fecnt) is list):
        srcdst[prefix + "feature_extractor_container"] = config_fe_r45_binormBIN_orig(3, feat_ch,cnt=fecnt,expf=expf,inplace=inplace,drop=drop);
    else:
        srcdst[prefix + "feature_extractor_container"] = config_fe_r45_binorm_orig(3, feat_ch,cnt=fecnt,expf=expf,inplace=inplace,drop=drop);
    srcdst[prefix + "feature_extractor_cco"] = config_bogo_resbinorm(prefix + "feature_extractor_container", "res1");
    srcdst[prefix + "feature_extractor_proto"] = config_bogo_resbinorm(prefix + "feature_extractor_container", "res2");
    return srcdst;

def arm_base_fe_moduleXL(srcdst, feat_ch, fecnt, prefix, inplace, drop=0, expf=1.5):

    srcdst[prefix + "feature_extractor_container"] = config_fe_r45_binorm_ptpt(3, feat_ch, cnt=fecnt, expf=expf,
                                                                               inplace=inplace, drop=drop);
    srcdst[prefix + "feature_extractor_cco"] = config_bogo_resbinorm(prefix + "feature_extractor_container",
                                                                     "res1");
    srcdst[prefix + "feature_extractor_proto"] = config_bogo_resbinorm(prefix + "feature_extractor_container",
                                                                       "res2");
    return srcdst;

def arm_base_fe_moduleXLIN(srcdst, feat_ch, fecnt, prefix, inplace, drop=0, expf=1.5):

    srcdst[prefix + "feature_extractor_container"] = config_fe_r45_binormBIN_ptpt(3, feat_ch, cnt=fecnt, expf=expf,
                                                                               inplace=inplace, drop=drop);
    srcdst[prefix + "feature_extractor_cco"] = config_bogo_resbinorm(prefix + "feature_extractor_container",
                                                                     "res1");
    srcdst[prefix + "feature_extractor_proto"] = config_bogo_resbinorm(prefix + "feature_extractor_container",
                                                                       "res2");
    return srcdst;

def arm_base_module_set(srcdst,prefix,maxT,capacity,feat_ch,tr_meta_path,expf=1,fecnt=3,wemb=0.3,wrej=0.1,inplace=True,detached_ta=False,detached_ga_proto=False,dropf=0,dropp=None):
    srcdst=arm_base_fe_module(srcdst,feat_ch,fecnt,prefix,inplace,dropf,expf=expf);
    srcdst=arm_common_part(srcdst, prefix, maxT, capacity, feat_ch, tr_meta_path, expf, fecnt, wemb, wrej,inplace,detached_ta=detached_ta,detached_ga_proto=detached_ga_proto,dropp=dropp);
    return srcdst;

def arm_base_module_setXL(srcdst,prefix,maxT,capacity,feat_ch,tr_meta_path,expf=1.5,fecnt=3,wemb=0.3,wrej=0.1,inplace=True,detached_ta=False,detached_ga_proto=False,dropf=0,dropp=None):
    srcdst=arm_base_fe_moduleXL(srcdst,feat_ch,fecnt,prefix,inplace,dropf,expf=expf);
    TAS= [
        [int(64*expf), 16, 64],
        [int(256*expf), 8, 32],
        [int(feat_ch), 8, 32]];
    srcdst=arm_common_part(srcdst, prefix, maxT, capacity, feat_ch, tr_meta_path, expf, fecnt, wemb, wrej,inplace,TA_scales=TAS,detached_ta=detached_ta,detached_ga_proto=detached_ga_proto,dropp=dropp);
    return srcdst;


def arm_base_module_setXLIN(srcdst,prefix,maxT,capacity,feat_ch,tr_meta_path,expf=1.5,fecnt=[3,1],wemb=0.3,wrej=0.1,inplace=True,detached_ta=False,detached_ga_proto=False,dropf=0,dropp=None):
    srcdst=arm_base_fe_moduleXLIN(srcdst,feat_ch,fecnt,prefix,inplace,dropf,expf=expf);
    TAS= [
        [int(64*expf), 16, 64],
        [int(256*expf), 8, 32],
        [int(feat_ch), 8, 32]];
    srcdst=arm_common_part(srcdst, prefix, maxT, capacity, feat_ch, tr_meta_path, expf, fecnt, wemb, wrej,inplace,TA_scales=TAS,detached_ta=detached_ta,detached_ga_proto=detached_ga_proto,dropp=dropp);
    return srcdst;
