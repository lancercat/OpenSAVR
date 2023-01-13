#
#
# from neko_2022_soai_zero.configs.modules.config_softlink import config_softlink
# from neko_2022_soai_zero.configs.modules.config_shuf import config_partial_part_sim_loss
# from neko_2022_soai_zero.configs.modules.config_shuf import config_chunk_shuf,config_chunk_shufA
#
# from neko_2022_soai_zero.configs.modules.config_recon import config_dcganN,config_masked_l2,config_masked_l2_reduction,\
#     config_masked_l2_reduction_gen2
from neko_2021_mjt.configs.common_subs.arm_post_fe_shared_prototyper import arm_shared_prototyper_np
from neko_2021_mjt.configs.bogo_modules.config_res_binorm import config_bogo_resbinorm;
from neko_2022_soai_zero.configs.modules.config_recon import config_masked_l2
from neko_2022_soai_zero.configs.modules.config_shuf import config_chunk_shuf
def arm_protorec_v2S(srcdst,prefix,feat_ch,speed=1):
    srcdst[prefix+"p_recon_loss"]=config_masked_l2(speed);
    return srcdst;

def arm_v2_shufS(srcdst,prefix,feat_ch,speed=1,rchunk=2,cchunk=2):
    srcdst[prefix+"shuf_img"]=config_chunk_shuf(rchunk,cchunk);
    srcdst[prefix+"shuf_recon_loss"]=config_masked_l2(speed=speed);
    return srcdst;

def arm_v2_shufDPS(srcdst,prefix,feat_ch,speed=1,rchunk=2,cchunk=2):
    srcdst[prefix + "feature_extractor_shuf_proto"] = config_bogo_resbinorm(prefix + "feature_extractor_container",
                                                                            "res3");
    srcdst = arm_shared_prototyper_np(
        srcdst, prefix, 512, feat_ch,
        prefix + "feature_extractor_shuf_proto",
        prefix + "GA",
        use_sp=False,
        detached_ga=False,
        drop=False,
        nameoverride=prefix + "shuf_proto"
    );
    srcdst=arm_v2_shufS(srcdst,prefix,feat_ch,speed,rchunk,cchunk);
    return srcdst;
#
# def arm_protorec_v2(srcdst,prefix,feat_ch,speed=1):
#     srcdst[prefix+"p_recon"]=config_dcganN(feat_ch);
#     srcdst[prefix+"p_recon_loss"]=config_masked_l2(speed);
#     return srcdst;
# def arm_fpmrec_v2(srcdst,prefix,feat_ch,speed=1):
#     srcdst[prefix+"f_recon"]=config_dcganN(feat_ch);
#     srcdst[prefix+"fpm_recon_loss"]=config_masked_l2_reduction(speed);
#     return srcdst;
# def arm_fpmrecS_v2(srcdst,prefix,feat_ch,speed=1):
#     srcdst[prefix+"fpm_recon_loss"]=config_masked_l2_reduction(speed);
#     return srcdst;
#
#
# def arm_v2_shuf(srcdst,prefix,feat_ch,speed=1,rchunk=2,cchunk=2):
#     srcdst[prefix+"shuf_recon"]=config_dcganN(feat_ch);
#     srcdst[prefix+"shuf_proto"]=config_softlink(prefix+"prototyper")
#     srcdst[prefix+"shuf_img"]=config_chunk_shuf(rchunk,cchunk);
#     srcdst[prefix+"shuf_recon_loss"]=config_masked_l2(speed=speed);
#     return srcdst;
# def arm_v2_shufDP(srcdst,prefix,feat_ch,speed=1,rchunk=2,cchunk=2):
#     srcdst[prefix+"shuf_recon"]=config_dcganN(feat_ch);
#     srcdst[prefix + "feature_extractor_shuf_proto"] = config_bogo_resbinorm(prefix + "feature_extractor_container", "res3");
#     srcdst = arm_shared_prototyper_np(
#         srcdst, prefix, 512, feat_ch,
#         prefix + "feature_extractor_shuf_proto",
#         prefix + "GA",
#         use_sp=False,
#         detached_ga=False,
#         drop=False,
#         nameoverride=prefix+"shuf_proto"
#     );
#     srcdst[prefix+"shuf_img"]=config_chunk_shuf(rchunk,cchunk);
#     srcdst[prefix+"shuf_recon_loss"]=config_masked_l2(speed=speed);
#     return srcdst;
#
# def arm_v2_shufS(srcdst,prefix,feat_ch,speed=1,rchunk=2,cchunk=2):
#     srcdst[prefix+"shuf_img"]=config_chunk_shuf(rchunk,cchunk);
#     srcdst[prefix+"shuf_recon_loss"]=config_masked_l2(speed=speed);
#     return srcdst;
#
#
# def arm_v2_shufSA(srcdst,prefix,feat_ch,speed=1,rchunk=2,cchunk=2):
#     srcdst[prefix+"shuf_img"]=config_chunk_shufA(rchunk,cchunk);
#     srcdst[prefix+"shuf_recon_loss"]=config_masked_l2(speed=speed);
#     return srcdst;
#
# def arm_v2_shufDPS(srcdst,prefix,feat_ch,speed=1,rchunk=2,cchunk=2):
#     srcdst[prefix + "feature_extractor_shuf_proto"] = config_bogo_resbinorm(prefix + "feature_extractor_container",
#                                                                             "res3");
#     srcdst = arm_shared_prototyper_np(
#         srcdst, prefix, 512, feat_ch,
#         prefix + "feature_extractor_shuf_proto",
#         prefix + "GA",
#         use_sp=False,
#         detached_ga=False,
#         drop=False,
#         nameoverride=prefix + "shuf_proto"
#     );
#     srcdst=arm_v2_shufS(srcdst,prefix,feat_ch,speed,rchunk,cchunk);
#     return srcdst;
# def arm_v2_shufADPS(srcdst,prefix,feat_ch,speed=1,rchunk=2,cchunk=2):
#     srcdst[prefix + "feature_extractor_shuf_proto"] = config_bogo_resbinorm(prefix + "feature_extractor_container",
#                                                                             "res3");
#     srcdst = arm_shared_prototyper_np(
#         srcdst, prefix, 512, feat_ch,
#         prefix + "feature_extractor_shuf_proto",
#         prefix + "GA",
#         use_sp=False,
#         detached_ga=False,
#         drop=False,
#         nameoverride=prefix + "shuf_proto"
#     );
#     srcdst=arm_v2_shufSA(srcdst,prefix,feat_ch,speed,rchunk,cchunk);
#     return srcdst;
#
# def arm_v2_frec(srcdst,prefix,speed=1):
#     srcdst[prefix+"fpm_recon_loss"]=config_masked_l2_reduction_gen2(speed);
#     return srcdst;
#
#
# def arm_shuf_partsim(srcdst,prefix,local_fcnt,pcnts,factor=16):
#     srcdst[prefix+"shuf_sim_loss"]=config_partial_part_sim_loss(local_fcnt,pcnts,factor);
#     return srcdst;
#
#
# def arm_v2_shufSPS(srcdst,prefix,feat_ch,speed=1,rchunk=2,cchunk=2):
#     srcdst[prefix + "shuf_recon"] = config_softlink(prefix + "p_recon");
#     srcdst=arm_v2_shufS(srcdst,prefix,feat_ch,speed,rchunk,cchunk);
#     return srcdst;
#
#
# def arm_v2_shuf_link(srcdst,prefix,feat_ch,speed=1):
#     srcdst[prefix+"shuf_img"]=config_chunk_shuf(2,2);
#     srcdst[prefix+"shuf_recon"]=config_softlink(prefix+"p_recon");
#     srcdst[prefix+"shuf_recon_loss"]=config_masked_l2(speed=speed);
#     return srcdst;
