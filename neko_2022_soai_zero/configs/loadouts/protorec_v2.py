
from neko_2022_soai_zero.configs.loadouts.base_model import arm_base_module_setXL,arm_base_module_setXLIN

from neko_2022_soai_zero.configs.loadouts.base_model import arm_base_module_set
# from neko_2022_soai_zero.configs.modules.config_bogo_partrec_loss import config_part_recon_loss
# from neko_2022_soai_zero.configs.modules.config_recon import config_dcganN,config_dcganN_insnorm,config_masked_l2,config_masked_l2_reduction,config_masked_l2_reduction_gen2
# from neko_2022_soai_zero.configs.loadouts.domd import arm_dom_mix,arm_dom_mix_slacky,arm_dom_mix_slacky_mva,arm_dom_mix_mva
# from neko_2022_soai_zero.configs.loadouts.protorec_cyc import arm_cycNS,arm_cycNSsp,arm_cycNsp,arm_cycN,arm_cyc_fe,arm_cyc_loss,arm_cyc_loss_o
# from neko_2022_soai_zero.configs.loadouts.individual_recs import arm_protorec_v2S,arm_protorec_v2,\
#     arm_v2_shufDPS,arm_v2_shufADPS,arm_shuf_partsim,arm_v2_shuf,arm_v2_shufDP,arm_v2_shufSPS,arm_v2_shuf_link,arm_v2_shufS,\
#     arm_v2_frec,arm_fpmrecS_v2,arm_fpmrec_v2

from neko_2022_soai_zero.configs.modules.config_recon import config_dcganN_insnorm
from neko_2022_soai_zero.configs.loadouts.individual_recs import arm_protorec_v2S,arm_v2_shufDPS
from neko_2022_soai_zero.configs.loadouts.protorec_cyc import arm_cyc_fe,arm_cyc_loss
def arm_protorec_v2IN_module_set(srcdst,prefix,maxT,capacity,feat_ch,tr_meta_path,expf=1,fecnt=3,wemb=0.3,wrej=0.1,inplace=True,detached_ta=False,detached_ga_proto=False,dropf=0,dropp=None,recon_speed=1):
    srcdst=arm_base_module_set(srcdst,prefix,maxT,capacity,feat_ch,tr_meta_path,expf,fecnt,wemb,wrej,inplace,detached_ta=detached_ta,detached_ga_proto=detached_ga_proto,dropf=dropf,dropp=dropp);
    srcdst[prefix + "p_recon"] = config_dcganN_insnorm(feat_ch);
    srcdst=arm_protorec_v2S(srcdst,prefix,feat_ch,speed=recon_speed);
    return srcdst;
def arm_v2IN_prec_module_setS(srcdst,prefix,maxT,capacity,feat_ch,tr_meta_path,expf=1,fecnt=3,wemb=0.3,wrej=0.1,inplace=True,detached_ta=False,detached_ga_proto=False,dropf=0,dropp=None,recon_speed=1,shuf_speed=1,rchunk=2,cchunk=2):
    srcdst=arm_base_module_set(srcdst,prefix,maxT,capacity,feat_ch,tr_meta_path,expf,fecnt,wemb,wrej,inplace,detached_ta=detached_ta,detached_ga_proto=detached_ga_proto,dropf=dropf,dropp=dropp);
    srcdst[prefix+"p_recon"]=config_dcganN_insnorm(feat_ch);
    srcdst=arm_protorec_v2S(srcdst,prefix,feat_ch,speed=recon_speed);
    return srcdst;
def arm_v2IN_prec_shuf_module_setDPS(srcdst,prefix,maxT,capacity,feat_ch,tr_meta_path,expf=1,fecnt=3,wemb=0.3,wrej=0.1,inplace=True,detached_ta=False,detached_ga_proto=False,dropf=0,dropp=None,recon_speed=1,shuf_speed=1,rchunk=2,cchunk=2):
    srcdst=arm_base_module_set(srcdst,prefix,maxT,capacity,feat_ch,tr_meta_path,expf,fecnt,wemb,wrej,inplace,detached_ta=detached_ta,detached_ga_proto=detached_ga_proto,dropf=dropf,dropp=dropp);
    srcdst[prefix+"p_recon"]=config_dcganN_insnorm(feat_ch);
    srcdst=arm_protorec_v2S(srcdst,prefix,feat_ch,speed=recon_speed);
    srcdst=arm_v2_shufDPS(srcdst,prefix,feat_ch,speed=shuf_speed,rchunk=rchunk,cchunk=cchunk);
    return srcdst;

def arm_v2INXL_prec_shuf_module_setDPS_cyc3_core(srcdst,prefix,maxT,capacity,feat_ch,tr_meta_path,expf=1.5,fecnt=3,wemb=0.3,wrej=0.1,inplace=True,detached_ta=False,detached_ga_proto=False,dropf=0,dropp=None,recon_speed=1,shuf_speed=1,rchunk=2,cchunk=2,cycspeed=1,cycaa=False):
    srcdst = arm_base_module_setXL(srcdst, prefix, maxT, capacity, feat_ch, tr_meta_path, expf, fecnt+1, wemb, wrej,
                                   inplace, detached_ta=detached_ta, detached_ga_proto=detached_ga_proto, dropf=dropf,
                                   dropp=dropp);
    return srcdst;
def arm_v2INXL_prec_shuf_module_setDPS_cyc3(srcdst,prefix,maxT,capacity,feat_ch,tr_meta_path,expf=1.5,fecnt=3,wemb=0.3,wrej=0.1,inplace=True,detached_ta=False,detached_ga_proto=False,dropf=0,dropp=None,recon_speed=1,shuf_speed=1,rchunk=2,cchunk=2,cycspeed=1,cycaa=False):
    srcdst = arm_base_module_setXL(srcdst, prefix, maxT, capacity, feat_ch, tr_meta_path, expf, fecnt+1, wemb, wrej,
                                   inplace, detached_ta=detached_ta, detached_ga_proto=detached_ga_proto, dropf=dropf,
                                   dropp=dropp);
    srcdst[prefix + "p_recon"] = config_dcganN_insnorm(feat_ch);
    srcdst = arm_protorec_v2S(srcdst, prefix, feat_ch, speed=recon_speed);
    srcdst = arm_v2_shufDPS(srcdst, prefix, feat_ch, speed=shuf_speed, rchunk=rchunk, cchunk=cchunk);
    srcdst=arm_cyc_fe(srcdst,prefix,feat_ch,"res4",feat_ch,aa=cycaa);
    srcdst=arm_cyc_loss(srcdst,prefix,wcls=cycspeed);
    return srcdst;

def arm_v2IN_prec_shuf_module_setDPS_cyc3(srcdst,prefix,maxT,capacity,feat_ch,tr_meta_path,expf=1,fecnt=3,wemb=0.3,wrej=0.1,inplace=True,detached_ta=False,detached_ga_proto=False,dropf=0,dropp=None,recon_speed=1,shuf_speed=1,rchunk=2,cchunk=2,cycspeed=1,cycaa=False):
    srcdst=arm_base_module_set(srcdst,prefix,maxT,capacity,feat_ch,tr_meta_path,expf,fecnt+1,wemb,wrej,inplace,detached_ta=detached_ta,detached_ga_proto=detached_ga_proto,dropf=dropf,dropp=dropp);
    srcdst[prefix+"p_recon"]=config_dcganN_insnorm(feat_ch);
    srcdst=arm_protorec_v2S(srcdst,prefix,feat_ch,speed=recon_speed);
    srcdst=arm_v2_shufDPS(srcdst,prefix,feat_ch,speed=shuf_speed,rchunk=rchunk,cchunk=cchunk);
    srcdst=arm_cyc_fe(srcdst,prefix,feat_ch,"res4",feat_ch,aa=cycaa);
    srcdst=arm_cyc_loss(srcdst,prefix,wcls=cycspeed);
    return srcdst;

#
# def arm_protorec_v2_shuf(srcdst,prefix,feat_ch,rec_speed,shuf_speed):
#     srcdst=arm_protorec_v2(srcdst,prefix,feat_ch,speed=rec_speed);
#     srcdst=arm_v2_shuf_link(srcdst,prefix,feat_ch,speed=shuf_speed);
#     return srcdst;
# def arm_protorec_v2_shufDP(srcdst,prefix,feat_ch,rec_speed,shuf_speed):
#     srcdst=arm_protorec_v2(srcdst,prefix,feat_ch,speed=rec_speed);
#     srcdst=arm_v2_shufDP(srcdst,prefix,feat_ch,speed=shuf_speed);
#     return srcdst;
# def arm_protorec_v2_shufDR(srcdst,prefix,feat_ch,rec_speed,shuf_speed):
#     srcdst=arm_protorec_v2(srcdst,prefix,feat_ch,speed=rec_speed);
#     srcdst=arm_v2_shuf(srcdst,prefix,feat_ch,speed=shuf_speed);
#     return srcdst;
# def arm_protorec_v2_shufS(srcdst,prefix,feat_ch,rec_speed,shuf_speed):
#     srcdst=arm_protorec_v2(srcdst,prefix,feat_ch,speed=rec_speed);
#     srcdst=arm_v2_shufS(srcdst,prefix,feat_ch,speed=shuf_speed);
#     return srcdst;
#
#
#
# def arm_protorec_v2_shuf_FS_module_set(srcdst,prefix,maxT,capacity,feat_ch,tr_meta_path,expf=1,fecnt=3,wemb=0.3,wrej=0.1,inplace=True,detached_ta=False,detached_ga_proto=False,dropf=0,dropp=None):
#     srcdst=arm_base_module_set(srcdst,prefix,maxT,capacity,feat_ch,tr_meta_path,expf,fecnt,wemb,wrej,inplace,detached_ta=detached_ta,detached_ga_proto=detached_ga_proto,dropf=dropf,dropp=dropp);
#     srcdst=arm_protorec_v2_shuf(srcdst,prefix,feat_ch);
#     srcdst=arm_dom_mix_slacky(srcdst,prefix,feat_ch,1.0);
#     return srcdst;
# def arm_protorec_v2_shuf_module_set(srcdst,prefix,maxT,capacity,feat_ch,tr_meta_path,expf=1,fecnt=3,wemb=0.3,wrej=0.1,inplace=True,detached_ta=False,detached_ga_proto=False,dropf=0,dropp=None,shuf_speed=1,rec_speed=1):
#     srcdst=arm_base_module_set(srcdst,prefix,maxT,capacity,feat_ch,tr_meta_path,expf,fecnt,wemb,wrej,inplace,detached_ta=detached_ta,detached_ga_proto=detached_ga_proto,dropf=dropf,dropp=dropp);
#     srcdst=arm_protorec_v2_shuf(srcdst,prefix,feat_ch,rec_speed=rec_speed,shuf_speed=shuf_speed);
#     return srcdst;
# def arm_protorec_v2_shuf_module_setDP(srcdst,prefix,maxT,capacity,feat_ch,tr_meta_path,expf=1,fecnt=3,wemb=0.3,wrej=0.1,inplace=True,detached_ta=False,detached_ga_proto=False,dropf=0,dropp=None,shuf_speed=1,rec_speed=1):
#     srcdst=arm_base_module_set(srcdst,prefix,maxT,capacity,feat_ch,tr_meta_path,expf,fecnt,wemb,wrej,inplace,detached_ta=detached_ta,detached_ga_proto=detached_ga_proto,dropf=dropf,dropp=dropp);
#     srcdst=arm_protorec_v2_shufDP(srcdst,prefix,feat_ch,rec_speed=rec_speed,shuf_speed=shuf_speed);
#     return srcdst;
#
# def arm_protorec_v2_shufXL_module_setDP(srcdst,prefix,maxT,capacity,feat_ch,tr_meta_path,expf=1.5,fecnt=3,wemb=0.3,wrej=0.1,inplace=True,detached_ta=False,detached_ga_proto=False,dropf=0,dropp=None,shuf_speed=1,rec_speed=1):
#     srcdst=arm_base_module_setXL(srcdst,prefix,maxT,capacity,feat_ch,tr_meta_path,expf,fecnt,wemb,wrej,inplace,detached_ta=detached_ta,detached_ga_proto=detached_ga_proto,dropf=dropf,dropp=dropp);
#     srcdst=arm_protorec_v2_shufDP(srcdst,prefix,feat_ch,rec_speed=rec_speed,shuf_speed=shuf_speed);
#     return srcdst;
#
# def arm_protorec_v2_shufS_module_set(srcdst,prefix,maxT,capacity,feat_ch,tr_meta_path,expf=1,fecnt=3,wemb=0.3,wrej=0.1,inplace=True,detached_ta=False,detached_ga_proto=False,dropf=0,dropp=None,shuf_speed=1,rec_speed=1):
#     srcdst=arm_base_module_set(srcdst,prefix,maxT,capacity,feat_ch,tr_meta_path,expf,fecnt,wemb,wrej,inplace,detached_ta=detached_ta,detached_ga_proto=detached_ga_proto,dropf=dropf,dropp=dropp);
#     srcdst=arm_protorec_v2_shufS(srcdst,prefix,feat_ch,rec_speed=rec_speed,shuf_speed=shuf_speed);
#     return srcdst;
#
#
# def arm_protorec_v2_shufSXL_module_set(srcdst, prefix, maxT, capacity, feat_ch, tr_meta_path, expf=1.5, fecnt=3,
#                                      wemb=0.3, wrej=0.1, inplace=True, detached_ta=False, detached_ga_proto=False,
#                                      dropf=0, dropp=None, shuf_speed=1, rec_speed=1):
#     srcdst=arm_base_module_setXL(srcdst,prefix,maxT,capacity,feat_ch,tr_meta_path,expf,fecnt,wemb,wrej,inplace,detached_ta=detached_ta,detached_ga_proto=detached_ga_proto,dropf=dropf,dropp=dropp);
#     srcdst = arm_protorec_v2_shufS(srcdst, prefix, feat_ch, rec_speed=rec_speed, shuf_speed=shuf_speed);
#     return srcdst;
#
#
# def arm_protorec_v2_shufDR_module_set(srcdst,prefix,maxT,capacity,feat_ch,tr_meta_path,expf=1,fecnt=3,wemb=0.3,wrej=0.1,inplace=True,detached_ta=False,detached_ga_proto=False,dropf=0,dropp=None,shuf_speed=1,rec_speed=1):
#     srcdst=arm_base_module_set(srcdst,prefix,maxT,capacity,feat_ch,tr_meta_path,expf,fecnt,wemb,wrej,inplace,detached_ta=detached_ta,detached_ga_proto=detached_ga_proto,dropf=dropf,dropp=dropp);
#     srcdst=arm_protorec_v2_shufDR(srcdst,prefix,feat_ch,shuf_speed=shuf_speed,rec_speed=rec_speed);
#     return srcdst;
#
#
# def arm_protorec_v2_shufDR_pps_module_set(srcdst,prefix,maxT,capacity,feat_ch,tr_meta_path,expf=1,fecnt=3,wemb=0.3,wrej=0.1,inplace=True,detached_ta=False,detached_ga_proto=False,dropf=0,dropp=None):
#     srcdst=arm_base_module_set(srcdst,prefix,maxT,capacity,feat_ch,tr_meta_path,expf,fecnt,wemb,wrej,inplace,detached_ta=detached_ta,detached_ga_proto=detached_ga_proto,dropf=dropf,dropp=dropp);
#     srcdst=arm_protorec_v2_shufDR(srcdst,prefix,feat_ch);
#     srcdst=arm_shuf_partsim(srcdst,prefix,256,4);
#     return srcdst;
#
#
#
# def arm_v2_shuf_module_set_mva(srcdst,prefix,maxT,capacity,feat_ch,tr_meta_path,expf=1,fecnt=3,wemb=0.3,wrej=0.1,inplace=True,detached_ta=False,detached_ga_proto=False,dropf=0,dropp=None,mva_speed=1):
#     srcdst=arm_base_module_set(srcdst,prefix,maxT,capacity,feat_ch,tr_meta_path,expf,fecnt,wemb,wrej,inplace,detached_ta=detached_ta,detached_ga_proto=detached_ga_proto,dropf=dropf,dropp=dropp);
#     srcdst=arm_v2_shuf(srcdst,prefix,feat_ch);
#     srcdst=arm_dom_mix_mva(srcdst,prefix,speed=mva_speed);
#     return srcdst;
#
# def arm_v2_shuf_module_set(srcdst,prefix,maxT,capacity,feat_ch,tr_meta_path,expf=1,fecnt=3,wemb=0.3,wrej=0.1,inplace=True,detached_ta=False,detached_ga_proto=False,dropf=0,dropp=None,shuf_speed=1,rchunk=2,cchunk=2):
#     srcdst=arm_base_module_set(srcdst,prefix,maxT,capacity,feat_ch,tr_meta_path,expf,fecnt,wemb,wrej,inplace,detached_ta=detached_ta,detached_ga_proto=detached_ga_proto,dropf=dropf,dropp=dropp);
#     srcdst=arm_v2_shuf(srcdst,prefix,feat_ch,speed=shuf_speed,rchunk=rchunk,cchunk=cchunk);
#     return srcdst;
#
# def arm_v2_shuf_module_setDP(srcdst,prefix,maxT,capacity,feat_ch,tr_meta_path,expf=1,fecnt=3,wemb=0.3,wrej=0.1,inplace=True,detached_ta=False,detached_ga_proto=False,dropf=0,dropp=None,shuf_speed=1,rchunk=2,cchunk=2):
#     srcdst=arm_base_module_set(srcdst,prefix,maxT,capacity,feat_ch,tr_meta_path,expf,fecnt,wemb,wrej,inplace,detached_ta=detached_ta,detached_ga_proto=detached_ga_proto,dropf=dropf,dropp=dropp);
#     srcdst=arm_v2_shufDP(srcdst,prefix,feat_ch,speed=shuf_speed,rchunk=rchunk,cchunk=cchunk);
#     return srcdst;
#
#
# def arm_v2_shufS_module_set(srcdst,prefix,maxT,capacity,feat_ch,tr_meta_path,expf=1,fecnt=3,wemb=0.3,wrej=0.1,inplace=True,detached_ta=False,detached_ga_proto=False,dropf=0,dropp=None):
#     srcdst=arm_base_module_setS(srcdst,prefix,maxT,capacity,feat_ch,tr_meta_path,expf,fecnt,wemb,wrej,inplace,detached_ta=detached_ta,detached_ga_proto=detached_ga_proto,dropf=dropf,dropp=dropp);
#     srcdst=arm_v2_shuf(srcdst,prefix,feat_ch);
#     return srcdst;
# def arm_v2_shuf_pps_module_set(srcdst,prefix,maxT,capacity,feat_ch,tr_meta_path,expf=1,fecnt=3,wemb=0.3,wrej=0.1,inplace=True,detached_ta=False,detached_ga_proto=False,dropf=0,dropp=None,ppsfac=16,shuf_speed=1,rchunk=2,cchunk=2):
#     srcdst=arm_base_module_set(srcdst,prefix,maxT,capacity,feat_ch,tr_meta_path,expf,fecnt,wemb,wrej,inplace,detached_ta=detached_ta,detached_ga_proto=detached_ga_proto,dropf=dropf,dropp=dropp);
#     srcdst=arm_v2_shuf(srcdst,prefix,feat_ch,shuf_speed,rchunk=rchunk,cchunk=cchunk);
#     srcdst=arm_shuf_partsim(srcdst,prefix,rchunk*cchunk*32,rchunk*cchunk,factor=ppsfac);
#     return srcdst;
# def arm_v2_shuf_pps_module_setDP(srcdst,prefix,maxT,capacity,feat_ch,tr_meta_path,expf=1,fecnt=3,wemb=0.3,wrej=0.1,inplace=True,detached_ta=False,detached_ga_proto=False,dropf=0,dropp=None,ppsfac=16,shuf_speed=1,rchunk=2,cchunk=2):
#     srcdst=arm_base_module_set(srcdst,prefix,maxT,capacity,feat_ch,tr_meta_path,expf,fecnt,wemb,wrej,inplace,detached_ta=detached_ta,detached_ga_proto=detached_ga_proto,dropf=dropf,dropp=dropp);
#     srcdst=arm_v2_shufDP(srcdst,prefix,feat_ch,shuf_speed,rchunk=rchunk,cchunk=cchunk);
#     srcdst=arm_shuf_partsim(srcdst,prefix,rchunk*cchunk*32,rchunk*cchunk,factor=ppsfac);
#     return srcdst;
#
# def arm_shuf_partrec(srcdst,prefix,rchunk,cchunk,lfcnt):
#     srcdst[prefix+"shuf_part_recon"]=config_dcganN(lfcnt//(rchunk*cchunk));
#     srcdst[prefix+"shuf_part_recon_loss_core"]=config_masked_l2();
#     srcdst[prefix+"shuf_part_recon_loss"]=\
#         config_part_recon_loss(recon_name=prefix+"shuf_part_recon",
#                                recon_loss_name=prefix+"shuf_part_recon_loss_core",
#                                rchunk=rchunk,cchunk=cchunk,local_fcnt=lfcnt);
#     return srcdst;
#
# def arm_v2_shuf_pre_module_set(srcdst,prefix,maxT,capacity,feat_ch,tr_meta_path,expf=1,fecnt=3,wemb=0.3,wrej=0.1,inplace=True,detached_ta=False,detached_ga_proto=False,dropf=0,dropp=None):
#     srcdst=arm_base_module_set(srcdst,prefix,maxT,capacity,feat_ch,tr_meta_path,expf,fecnt,wemb,wrej,inplace,detached_ta=detached_ta,detached_ga_proto=detached_ga_proto,dropf=dropf,dropp=dropp);
#     srcdst=arm_v2_shuf(srcdst,prefix,feat_ch);
#     srcdst=arm_shuf_partrec(srcdst,prefix,2,2,256);
#     return srcdst;
#
#
#
# def arm_v2XL_shuf_pps_module_set(srcdst,prefix,maxT,capacity,feat_ch,tr_meta_path,expf=1,fecnt=3,wemb=0.3,wrej=0.1,inplace=True,detached_ta=False,detached_ga_proto=False,dropf=0,dropp=None):
#     srcdst=arm_base_module_setXL(srcdst,prefix,maxT,capacity,feat_ch,tr_meta_path,expf,fecnt,wemb,wrej,inplace,detached_ta=detached_ta,detached_ga_proto=detached_ga_proto,dropf=dropf,dropp=dropp);
#     srcdst=arm_v2_shuf(srcdst,prefix,feat_ch);
#     srcdst=arm_shuf_partsim(srcdst,prefix,256,4);
#     return srcdst;
#
#
#
#
# def arm_protorec_v2_shuf_mva_module_set(srcdst,prefix,maxT,capacity,feat_ch,tr_meta_path,expf=1,fecnt=3,wemb=0.3,wrej=0.1,inplace=True,detached_ta=False,detached_ga_proto=False,dropf=0,dropp=None):
#     srcdst=arm_base_module_set(srcdst,prefix,maxT,capacity,feat_ch,tr_meta_path,expf,fecnt,wemb,wrej,inplace,detached_ta=detached_ta,detached_ga_proto=detached_ga_proto,dropf=dropf,dropp=dropp);
#     srcdst=arm_protorec_v2_shuf(srcdst,prefix,feat_ch);
#     srcdst=arm_dom_mix_slacky_mva(srcdst,prefix,feat_ch,speed=1);
#     return srcdst;
# def arm_protorec_v2_shuf_mva_only_module_set(srcdst,prefix,maxT,capacity,feat_ch,tr_meta_path,expf=1,fecnt=3,wemb=0.3,wrej=0.1,inplace=True,detached_ta=False,detached_ga_proto=False,dropf=0,dropp=None):
#     srcdst=arm_base_module_set(srcdst,prefix,maxT,capacity,feat_ch,tr_meta_path,expf,fecnt,wemb,wrej,inplace,detached_ta=detached_ta,detached_ga_proto=detached_ga_proto,dropf=dropf,dropp=dropp);
#     srcdst=arm_protorec_v2_shuf(srcdst,prefix,feat_ch);
#     srcdst=arm_dom_mix_mva(srcdst,prefix);
#     return srcdst;
# def arm_protorec_v2_shuf_ns3_module_set(srcdst,prefix,maxT,capacity,feat_ch,tr_meta_path,expf=1,fecnt=3,wemb=0.3,wrej=0.1,inplace=True,detached_ta=False,detached_ga_proto=False,dropf=0,dropp=None):
#     srcdst=arm_base_module_set(srcdst,prefix,maxT,capacity,feat_ch,tr_meta_path,expf,fecnt,wemb,wrej,inplace,detached_ta=detached_ta,detached_ga_proto=detached_ga_proto,dropf=dropf,dropp=dropp);
#     srcdst=arm_protorec_v2_shuf(srcdst,prefix,feat_ch);
#     srcdst=arm_cycNS(srcdst,prefix,feat_ch,detach_ga=detached_ga_proto)
#     return srcdst;
#
# def arm_protorec_v2_shuf_n3_module_set(srcdst,prefix,maxT,capacity,feat_ch,tr_meta_path,expf=1,fecnt=3,wemb=0.3,wrej=0.1,inplace=True,detached_ta=False,detached_ga_proto=False,dropf=0,dropp=None):
#     srcdst=arm_base_module_set(srcdst,prefix,maxT,capacity,feat_ch,tr_meta_path,expf,fecnt,wemb,wrej,inplace,detached_ta=detached_ta,detached_ga_proto=detached_ga_proto,dropf=dropf,dropp=dropp);
#     srcdst=arm_protorec_v2_shufDR(srcdst,prefix,feat_ch);
#     srcdst=arm_cycNS(srcdst,prefix,feat_ch,detach_ga=detached_ga_proto)
#     return srcdst;
#
# def arm_protorec_v2_shuf_n3sp_module_set(srcdst,prefix,maxT,capacity,feat_ch,tr_meta_path,expf=1,fecnt=3,wemb=0.3,wrej=0.1,inplace=True,detached_ta=False,detached_ga_proto=False,dropf=0,dropp=None):
#     srcdst=arm_base_module_set(srcdst,prefix,maxT,capacity,feat_ch,tr_meta_path,expf,fecnt,wemb,wrej,inplace,detached_ta=detached_ta,detached_ga_proto=detached_ga_proto,dropf=dropf,dropp=dropp);
#     srcdst=arm_protorec_v2_shufDR(srcdst,prefix,feat_ch);
#     srcdst=arm_cycNsp(srcdst,prefix,feat_ch,detach_ga=detached_ga_proto)
#     return srcdst;
# def arm_protorec_v2_shuf_ns3mvap_module_set(srcdst,prefix,maxT,capacity,feat_ch,tr_meta_path,expf=1,fecnt=3,wemb=0.3,wrej=0.1,inplace=True,detached_ta=False,detached_ga_proto=False,dropf=0,dropp=None):
#     srcdst=arm_base_module_set(srcdst,prefix,maxT,capacity,feat_ch,tr_meta_path,expf,fecnt,wemb,wrej,inplace,detached_ta=detached_ta,detached_ga_proto=detached_ga_proto,dropf=dropf,dropp=dropp);
#     srcdst=arm_protorec_v2_shuf(srcdst,prefix,feat_ch);
#     srcdst=arm_cycNS(srcdst,prefix,feat_ch,detach_ga=detached_ga_proto);
#     return srcdst;
# def arm_protorec_v2_shuf_ns3sp_module_set(srcdst,prefix,maxT,capacity,feat_ch,tr_meta_path,expf=1,fecnt=3,wemb=0.3,wrej=0.1,inplace=True,detached_ta=False,detached_ga_proto=False,dropf=0,dropp=None):
#     srcdst=arm_base_module_set(srcdst,prefix,maxT,capacity,feat_ch,tr_meta_path,expf,fecnt,wemb,wrej,inplace,detached_ta=detached_ta,detached_ga_proto=detached_ga_proto,dropf=dropf,dropp=dropp);
#     srcdst=arm_protorec_v2_shuf(srcdst,prefix,feat_ch);
#     srcdst=arm_cycNSsp(srcdst,prefix,feat_ch,detach_ga=detached_ga_proto)
#     return srcdst;
# def arm_protorec_v2_ns3sp_module_set(srcdst,prefix,maxT,capacity,feat_ch,tr_meta_path,expf=1,fecnt=3,wemb=0.3,wrej=0.1,inplace=True,detached_ta=False,detached_ga_proto=False,dropf=0,dropp=None):
#     srcdst=arm_base_module_set(srcdst,prefix,maxT,capacity,feat_ch,tr_meta_path,expf,fecnt,wemb,wrej,inplace,detached_ta=detached_ta,detached_ga_proto=detached_ga_proto,dropf=dropf,dropp=dropp);
#     srcdst=arm_protorec_v2(srcdst,prefix,feat_ch);
#     srcdst=arm_cycNSsp(srcdst,prefix,feat_ch,detach_ga=detached_ga_proto)
#     return srcdst;
#
# def arm_protorec_v2_shuf_ns3sp_module_setXL(srcdst,prefix,maxT,capacity,feat_ch,tr_meta_path,expf=1.5,fecnt=3,wemb=0.3,wrej=0.1,inplace=True,detached_ta=False,detached_ga_proto=False,dropf=0,dropp=None):
#     srcdst=arm_base_module_setXL(srcdst,prefix,maxT,capacity,feat_ch,tr_meta_path,expf,fecnt,wemb,wrej,inplace,detached_ta=detached_ta,detached_ga_proto=detached_ga_proto,dropf=dropf,dropp=dropp);
#     srcdst=arm_protorec_v2_shuf(srcdst,prefix,feat_ch);
#     srcdst=arm_cycNSsp(srcdst,prefix,feat_ch,detach_ga=detached_ga_proto)
#     return srcdst;
#
# def arm_protorec_v2_shuf_module_setXL(srcdst,prefix,maxT,capacity,feat_ch,tr_meta_path,expf=1.5,fecnt=3,wemb=0.3,wrej=0.1,inplace=True,detached_ta=False,detached_ga_proto=False,dropf=0,dropp=None):
#     srcdst=arm_base_module_setXL(srcdst,prefix,maxT,capacity,feat_ch,tr_meta_path,expf,fecnt,wemb,wrej,inplace,detached_ta=detached_ta,detached_ga_proto=detached_ga_proto,dropf=dropf,dropp=dropp);
#     srcdst=arm_protorec_v2_shuf(srcdst,prefix,feat_ch);
#     return srcdst;
#
# def arm_protorecS_v2_shuf_module_set(srcdst,prefix,maxT,capacity,feat_ch,tr_meta_path,expf=1,fecnt=3,wemb=0.3,wrej=0.1,inplace=True,detached_ta=False,detached_ga_proto=False,dropf=0,dropp=None):
#     srcdst=arm_base_module_setS(srcdst,prefix,maxT,capacity,feat_ch,tr_meta_path,expf,fecnt,wemb,wrej,inplace,detached_ta=detached_ta,detached_ga_proto=detached_ga_proto,dropf=dropf,dropp=dropp);
#     srcdst=arm_protorec_v2_shuf(srcdst,prefix,feat_ch);
#     return srcdst;
#
# def arm_protorecS_v2_shuf_ns3_module_set(srcdst,prefix,maxT,capacity,feat_ch,tr_meta_path,expf=1,fecnt=3,wemb=0.3,wrej=0.1,inplace=True,detached_ta=False,detached_ga_proto=False,dropf=0,dropp=None):
#     srcdst=arm_base_module_setS(srcdst,prefix,maxT,capacity,feat_ch,tr_meta_path,expf,fecnt,wemb,wrej,inplace,detached_ta=detached_ta,detached_ga_proto=detached_ga_proto,dropf=dropf,dropp=dropp);
#     srcdst=arm_protorec_v2_shuf(srcdst,prefix,feat_ch);
#     srcdst=arm_cycNS(srcdst,prefix,feat_ch,detach_ga=detached_ga_proto)
#     return srcdst;
#
# def arm_protorecS_v2_shuf_ns3_mva_module_set(srcdst,prefix,maxT,capacity,feat_ch,tr_meta_path,expf=1,fecnt=3,wemb=0.3,wrej=0.1,inplace=True,detached_ta=False,detached_ga_proto=False,dropf=0,dropp=None):
#     srcdst=arm_base_module_setS(srcdst,prefix,maxT,capacity,feat_ch,tr_meta_path,expf,fecnt,wemb,wrej,inplace,detached_ta=detached_ta,detached_ga_proto=detached_ga_proto,dropf=dropf,dropp=dropp);
#     srcdst=arm_protorec_v2_shuf(srcdst,prefix,feat_ch);
#     srcdst=arm_cycNS(srcdst,prefix,feat_ch,detach_ga=detached_ga_proto);
#     srcdst=arm_dom_mix_slacky_mva(srcdst,prefix,feat_ch,speed=0.1);
#     return srcdst;
#
#
# def arm_protorec_v2_module_set(srcdst,prefix,maxT,capacity,feat_ch,tr_meta_path,expf=1,fecnt=3,wemb=0.3,wrej=0.1,inplace=True,detached_ta=False,detached_ga_proto=False,dropf=0,dropp=None,recon_speed=1):
#     srcdst=arm_base_module_set(srcdst,prefix,maxT,capacity,feat_ch,tr_meta_path,expf,fecnt,wemb,wrej,inplace,detached_ta=detached_ta,detached_ga_proto=detached_ga_proto,dropf=dropf,dropp=dropp);
#     srcdst[prefix + "p_recon"] = config_dcganN(feat_ch);
#     srcdst=arm_protorec_v2S(srcdst,prefix,feat_ch,speed=recon_speed);
#     return srcdst;
#
# def arm_fpmrec_v2_module_set(srcdst,prefix,maxT,capacity,feat_ch,tr_meta_path,expf=1,fecnt=3,wemb=0.3,wrej=0.1,inplace=True,detached_ta=False,detached_ga_proto=False,dropf=0,dropp=None,fpm_speed=1):
#     srcdst=arm_base_module_set(srcdst,prefix,maxT,capacity,feat_ch,tr_meta_path,expf,fecnt,wemb,wrej,inplace,detached_ta=detached_ta,detached_ga_proto=detached_ga_proto,dropf=dropf,dropp=dropp);
#     srcdst[prefix+"p_recon"]=config_dcganN(feat_ch);
#     srcdst=arm_fpmrec_v2(srcdst,prefix,feat_ch,speed=fpm_speed);
#     return srcdst;
#
# def arm_fpmrec_protorec_v2_module_set(srcdst,prefix,maxT,capacity,feat_ch,tr_meta_path,expf=1,fecnt=3,wemb=0.3,wrej=0.1,inplace=True,detached_ta=False,detached_ga_proto=False,dropf=0,dropp=None,fpm_speed=1,recon_speed=1):
#     srcdst=arm_base_module_set(srcdst,prefix,maxT,capacity,feat_ch,tr_meta_path,expf,fecnt,wemb,wrej,inplace,detached_ta=detached_ta,detached_ga_proto=detached_ga_proto,dropf=dropf,dropp=dropp);
#     srcdst[prefix+"p_recon"]=config_dcganN(feat_ch);
#     srcdst=arm_fpmrec_v2(srcdst,prefix,feat_ch,speed=fpm_speed);
#     srcdst=arm_protorec_v2S(srcdst,prefix,feat_ch,speed=recon_speed);
#     return srcdst;
# def arm_protorec_v2_shufDPS_module_set(srcdst,prefix,maxT,capacity,feat_ch,tr_meta_path,expf=1,fecnt=3,wemb=0.3,wrej=0.1,inplace=True,detached_ta=False,detached_ga_proto=False,dropf=0,dropp=None,shuf_speed=1,rec_speed=1):
#     srcdst=arm_base_module_set(srcdst,prefix,maxT,capacity,feat_ch,tr_meta_path,expf,fecnt,wemb,wrej,inplace,detached_ta=detached_ta,detached_ga_proto=detached_ga_proto,dropf=dropf,dropp=dropp);
#     srcdst[prefix+"p_recon"]=config_dcganN(feat_ch);
#     srcdst=arm_v2_shufDPS(srcdst,prefix,feat_ch,speed=shuf_speed);
#     srcdst=arm_protorec_v2S(srcdst,prefix,feat_ch,speed=rec_speed);
#     return srcdst;
#
# def arm_v2_shufDPS_frecS_module_set(srcdst,prefix,maxT,capacity,feat_ch,tr_meta_path,expf=1,fecnt=3,wemb=0.3,wrej=0.1,inplace=True,detached_ta=False,detached_ga_proto=False,dropf=0,dropp=None,shuf_speed=1,rec_speed=1,fpm_speed=1):
#     srcdst=arm_base_module_set(srcdst,prefix,maxT,capacity,feat_ch,tr_meta_path,expf,fecnt,wemb,wrej,inplace,detached_ta=detached_ta,detached_ga_proto=detached_ga_proto,dropf=dropf,dropp=dropp);
#     srcdst[prefix+"p_recon"]=config_dcganN(feat_ch);
#     srcdst=arm_v2_shufDPS(srcdst,prefix,feat_ch,speed=shuf_speed);
#     srcdst=arm_fpmrecS_v2(srcdst,prefix,feat_ch,speed=fpm_speed);
#     return srcdst;
#
#
#
# def arm_v2_shuf_module_setDPS(srcdst,prefix,maxT,capacity,feat_ch,tr_meta_path,expf=1,fecnt=3,wemb=0.3,wrej=0.1,inplace=True,detached_ta=False,detached_ga_proto=False,dropf=0,dropp=None,shuf_speed=1,rchunk=2,cchunk=2):
#     srcdst=arm_base_module_set(srcdst,prefix,maxT,capacity,feat_ch,tr_meta_path,expf,fecnt,wemb,wrej,inplace,detached_ta=detached_ta,detached_ga_proto=detached_ga_proto,dropf=dropf,dropp=dropp);
#     srcdst[prefix+"p_recon"]=config_dcganN(feat_ch);
#     srcdst=arm_v2_shufDPS(srcdst,prefix,feat_ch,speed=shuf_speed,rchunk=rchunk,cchunk=cchunk);
#     return srcdst;
# def arm_protorec_v2_shufDPS_frecS_module_set(srcdst,prefix,maxT,capacity,feat_ch,tr_meta_path,expf=1,fecnt=3,wemb=0.3,wrej=0.1,inplace=True,detached_ta=False,detached_ga_proto=False,dropf=0,dropp=None,shuf_speed=1,rec_speed=1,fpm_speed=1):
#     srcdst=arm_base_module_set(srcdst,prefix,maxT,capacity,feat_ch,tr_meta_path,expf,fecnt,wemb,wrej,inplace,detached_ta=detached_ta,detached_ga_proto=detached_ga_proto,dropf=dropf,dropp=dropp);
#     srcdst[prefix+"p_recon"]=config_dcganN(feat_ch);
#     srcdst=arm_protorec_v2S(srcdst,prefix,feat_ch,speed=rec_speed);
#     srcdst=arm_v2_shufDPS(srcdst,prefix,feat_ch,speed=shuf_speed);
#     srcdst=arm_fpmrecS_v2(srcdst,prefix,feat_ch,speed=fpm_speed);
#     return srcdst;
#
#
# def arm_v2IN_shuf_module_setDPS(srcdst,prefix,maxT,capacity,feat_ch,tr_meta_path,expf=1,fecnt=3,wemb=0.3,wrej=0.1,inplace=True,detached_ta=False,detached_ga_proto=False,dropf=0,dropp=None,shuf_speed=1,rchunk=2,cchunk=2):
#     srcdst=arm_base_module_set(srcdst,prefix,maxT,capacity,feat_ch,tr_meta_path,expf,fecnt,wemb,wrej,inplace,detached_ta=detached_ta,detached_ga_proto=detached_ga_proto,dropf=dropf,dropp=dropp);
#     srcdst[prefix+"p_recon"]=config_dcganN_insnorm(feat_ch);
#     srcdst=arm_v2_shufDPS(srcdst,prefix,feat_ch,speed=shuf_speed,rchunk=rchunk,cchunk=cchunk);
#     return srcdst;
# def arm_v2IN_shufA_module_setDPS(srcdst,prefix,maxT,capacity,feat_ch,tr_meta_path,expf=1,fecnt=3,wemb=0.3,wrej=0.1,inplace=True,detached_ta=False,detached_ga_proto=False,dropf=0,dropp=None,shuf_speed=1,rchunk=2,cchunk=2):
#     srcdst=arm_base_module_set(srcdst,prefix,maxT,capacity,feat_ch,tr_meta_path,expf,fecnt,wemb,wrej,inplace,detached_ta=detached_ta,detached_ga_proto=detached_ga_proto,dropf=dropf,dropp=dropp);
#     srcdst[prefix+"p_recon"]=config_dcganN_insnorm(feat_ch);
#     srcdst=arm_v2_shufADPS(srcdst,prefix,feat_ch,speed=shuf_speed,rchunk=rchunk,cchunk=cchunk);
#     return srcdst;
# def arm_v2IN_shuf_module_setSPS(srcdst,prefix,maxT,capacity,feat_ch,tr_meta_path,expf=1,fecnt=3,wemb=0.3,wrej=0.1,inplace=True,detached_ta=False,detached_ga_proto=False,dropf=0,dropp=None,shuf_speed=1,rchunk=2,cchunk=2):
#     srcdst=arm_base_module_set(srcdst,prefix,maxT,capacity,feat_ch,tr_meta_path,expf,fecnt,wemb,wrej,inplace,detached_ta=detached_ta,detached_ga_proto=detached_ga_proto,dropf=dropf,dropp=dropp);
#     srcdst[prefix+"p_recon"]=config_dcganN_insnorm(feat_ch);
#     srcdst=arm_v2_shufSPS(srcdst,prefix,feat_ch,speed=shuf_speed,rchunk=rchunk,cchunk=cchunk);
#     return srcdst;
#
#
#
# def arm_v2IN_prec_shuf_module_setDPS(srcdst,prefix,maxT,capacity,feat_ch,tr_meta_path,expf=1,fecnt=3,wemb=0.3,wrej=0.1,inplace=True,detached_ta=False,detached_ga_proto=False,dropf=0,dropp=None,recon_speed=1,shuf_speed=1,rchunk=2,cchunk=2):
#     srcdst=arm_base_module_set(srcdst,prefix,maxT,capacity,feat_ch,tr_meta_path,expf,fecnt,wemb,wrej,inplace,detached_ta=detached_ta,detached_ga_proto=detached_ga_proto,dropf=dropf,dropp=dropp);
#     srcdst[prefix+"p_recon"]=config_dcganN_insnorm(feat_ch);
#     srcdst=arm_protorec_v2S(srcdst,prefix,feat_ch,speed=recon_speed);
#     srcdst=arm_v2_shufDPS(srcdst,prefix,feat_ch,speed=shuf_speed,rchunk=rchunk,cchunk=cchunk);
#     return srcdst;
#
# def arm_v2IN_prec_shuf_module_setDPS_mva(srcdst,prefix,maxT,capacity,feat_ch,tr_meta_path,expf=1,fecnt=3,wemb=0.3,wrej=0.1,inplace=True,detached_ta=False,detached_ga_proto=False,dropf=0,dropp=None,recon_speed=1,shuf_speed=1,daspeed=1,rchunk=2,cchunk=2):
#     srcdst=arm_base_module_set(srcdst,prefix,maxT,capacity,feat_ch,tr_meta_path,expf,fecnt,wemb,wrej,inplace,detached_ta=detached_ta,detached_ga_proto=detached_ga_proto,dropf=dropf,dropp=dropp);
#     srcdst[prefix+"p_recon"]=config_dcganN_insnorm(feat_ch);
#     srcdst=arm_protorec_v2S(srcdst,prefix,feat_ch,speed=recon_speed);
#     srcdst=arm_v2_shufDPS(srcdst,prefix,feat_ch,speed=shuf_speed,rchunk=rchunk,cchunk=cchunk);
#     srcdst=arm_dom_mix_mva(srcdst,prefix,daspeed)
#     return srcdst;
# def arm_v2IN_prec_shuf_module_setDPS_dis(srcdst,prefix,maxT,capacity,feat_ch,tr_meta_path,expf=1,fecnt=3,wemb=0.3,wrej=0.1,inplace=True,detached_ta=False,detached_ga_proto=False,dropf=0,dropp=None,recon_speed=1,shuf_speed=1,daspeed=1,rchunk=2,cchunk=2,domcnt=2):
#     srcdst=arm_base_module_set(srcdst,prefix,maxT,capacity,feat_ch,tr_meta_path,expf,fecnt,wemb,wrej,inplace,detached_ta=detached_ta,detached_ga_proto=detached_ga_proto,dropf=dropf,dropp=dropp);
#     srcdst[prefix+"p_recon"]=config_dcganN_insnorm(feat_ch);
#     srcdst=arm_protorec_v2S(srcdst,prefix,feat_ch,speed=recon_speed);
#     srcdst=arm_v2_shufDPS(srcdst,prefix,feat_ch,speed=shuf_speed,rchunk=rchunk,cchunk=cchunk);
#     srcdst=arm_dom_mix_slacky(srcdst,prefix,feat_ch,daspeed,domcnt=domcnt)
#     return srcdst;
# def arm_v2IN_prec_shufA_module_setDPS(srcdst,prefix,maxT,capacity,feat_ch,tr_meta_path,expf=1,fecnt=3,wemb=0.3,wrej=0.1,inplace=True,detached_ta=False,detached_ga_proto=False,dropf=0,dropp=None,recon_speed=1,shuf_speed=1,rchunk=2,cchunk=2):
#     srcdst=arm_base_module_set(srcdst,prefix,maxT,capacity,feat_ch,tr_meta_path,expf,fecnt,wemb,wrej,inplace,detached_ta=detached_ta,detached_ga_proto=detached_ga_proto,dropf=dropf,dropp=dropp);
#     srcdst[prefix+"p_recon"]=config_dcganN_insnorm(feat_ch);
#     srcdst=arm_protorec_v2S(srcdst,prefix,feat_ch,speed=recon_speed);
#     srcdst=arm_v2_shufADPS(srcdst,prefix,feat_ch,speed=shuf_speed,rchunk=rchunk,cchunk=cchunk);
#     return srcdst;
# def arm_v2INXL_prec_shuf_module_setDPS(srcdst,prefix,maxT,capacity,feat_ch,tr_meta_path,expf=1.5,fecnt=3,wemb=0.3,wrej=0.1,inplace=True,detached_ta=False,detached_ga_proto=False,dropf=0,dropp=None,recon_speed=1,shuf_speed=1,rchunk=2,cchunk=2):
#     srcdst=arm_base_module_setXL(srcdst,prefix,maxT,capacity,feat_ch,tr_meta_path,expf,fecnt,wemb,wrej,inplace,detached_ta=detached_ta,detached_ga_proto=detached_ga_proto,dropf=dropf,dropp=dropp);
#     srcdst[prefix+"p_recon"]=config_dcganN_insnorm(feat_ch);
#     srcdst=arm_protorec_v2S(srcdst,prefix,feat_ch,speed=recon_speed);
#     srcdst=arm_v2_shufDPS(srcdst,prefix,feat_ch,speed=shuf_speed,rchunk=rchunk,cchunk=cchunk);
#     return srcdst;
#
# def arm_v2INXL_prec_shuf_module_setDPS_dis(srcdst,prefix,maxT,capacity,feat_ch,tr_meta_path,expf=1.5,fecnt=3,wemb=0.3,wrej=0.1,inplace=True,detached_ta=False,detached_ga_proto=False,dropf=0,dropp=None,recon_speed=1,shuf_speed=1,rchunk=2,cchunk=2):
#     srcdst=arm_base_module_setXL(srcdst,prefix,maxT,capacity,feat_ch,tr_meta_path,expf,fecnt,wemb,wrej,inplace,detached_ta=detached_ta,detached_ga_proto=detached_ga_proto,dropf=dropf,dropp=dropp);
#     srcdst[prefix+"p_recon"]=config_dcganN_insnorm(feat_ch);
#     srcdst=arm_protorec_v2S(srcdst,prefix,feat_ch,speed=recon_speed);
#     srcdst=arm_v2_shufDPS(srcdst,prefix,feat_ch,speed=shuf_speed,rchunk=rchunk,cchunk=cchunk);
#     srcdst=arm_dom_mix_slacky(srcdst,prefix, feat_ch);
#     return srcdst;
# def arm_v2IN_prec_module_setS(srcdst,prefix,maxT,capacity,feat_ch,tr_meta_path,expf=1,fecnt=3,wemb=0.3,wrej=0.1,inplace=True,detached_ta=False,detached_ga_proto=False,dropf=0,dropp=None,recon_speed=1,shuf_speed=1,rchunk=2,cchunk=2):
#     srcdst=arm_base_module_set(srcdst,prefix,maxT,capacity,feat_ch,tr_meta_path,expf,fecnt,wemb,wrej,inplace,detached_ta=detached_ta,detached_ga_proto=detached_ga_proto,dropf=dropf,dropp=dropp);
#     srcdst[prefix+"p_recon"]=config_dcganN_insnorm(feat_ch);
#     srcdst=arm_protorec_v2S(srcdst,prefix,feat_ch,speed=recon_speed);
#     return srcdst;
# def arm_v2IN_prec_shuf_module_setSPS(srcdst,prefix,maxT,capacity,feat_ch,tr_meta_path,expf=1,fecnt=3,wemb=0.3,wrej=0.1,inplace=True,detached_ta=False,detached_ga_proto=False,dropf=0,dropp=None,recon_speed=1,shuf_speed=1,rchunk=2,cchunk=2):
#     srcdst=arm_base_module_set(srcdst,prefix,maxT,capacity,feat_ch,tr_meta_path,expf,fecnt,wemb,wrej,inplace,detached_ta=detached_ta,detached_ga_proto=detached_ga_proto,dropf=dropf,dropp=dropp);
#     srcdst[prefix+"p_recon"]=config_dcganN_insnorm(feat_ch);
#     srcdst=arm_protorec_v2S(srcdst,prefix,feat_ch,speed=recon_speed);
#     srcdst=arm_v2_shufSPS(srcdst,prefix,feat_ch,speed=shuf_speed,rchunk=rchunk,cchunk=cchunk);
#     return srcdst;
# def arm_v2IN_shuf_module_setDPS_cyc3(srcdst,prefix,maxT,capacity,feat_ch,tr_meta_path,expf=1,fecnt=3,wemb=0.3,wrej=0.1,inplace=True,detached_ta=False,detached_ga_proto=False,dropf=0,dropp=None,shuf_speed=1,rchunk=2,cchunk=2):
#     srcdst=arm_base_module_set(srcdst,prefix,maxT,capacity,feat_ch,tr_meta_path,expf,fecnt+1,wemb,wrej,inplace,detached_ta=detached_ta,detached_ga_proto=detached_ga_proto,dropf=dropf,dropp=dropp,);
#     srcdst[prefix+"p_recon"]=config_dcganN_insnorm(feat_ch);
#     srcdst=arm_cyc_fe(srcdst,prefix,feat_ch,"res4",feat_ch);
#     srcdst=arm_cyc_loss(srcdst,prefix);
#     srcdst=arm_v2_shufDPS(srcdst,prefix,feat_ch,speed=shuf_speed,rchunk=rchunk,cchunk=cchunk);
#     return srcdst;
#
#
# def arm_v2IN_prec_shufA_module_setDPS_cyc3(srcdst,prefix,maxT,capacity,feat_ch,tr_meta_path,expf=1,fecnt=3,wemb=0.3,wrej=0.1,inplace=True,detached_ta=False,detached_ga_proto=False,dropf=0,dropp=None,recon_speed=1,shuf_speed=1,rchunk=2,cchunk=2,cycspeed=1):
#     srcdst=arm_base_module_set(srcdst,prefix,maxT,capacity,feat_ch,tr_meta_path,expf,fecnt+1,wemb,wrej,inplace,detached_ta=detached_ta,detached_ga_proto=detached_ga_proto,dropf=dropf,dropp=dropp);
#     srcdst[prefix+"p_recon"]=config_dcganN_insnorm(feat_ch);
#     srcdst=arm_protorec_v2S(srcdst,prefix,feat_ch,speed=recon_speed);
#     srcdst=arm_v2_shufADPS(srcdst,prefix,feat_ch,speed=shuf_speed,rchunk=rchunk,cchunk=cchunk);
#     srcdst=arm_cyc_fe(srcdst,prefix,feat_ch,"res4",feat_ch);
#     srcdst=arm_cyc_loss(srcdst,prefix,wcls=cycspeed);
#     return srcdst;
#
# def arm_v2IN_prec_shuf_module_setDPS_cyc3(srcdst,prefix,maxT,capacity,feat_ch,tr_meta_path,expf=1,fecnt=3,wemb=0.3,wrej=0.1,inplace=True,detached_ta=False,detached_ga_proto=False,dropf=0,dropp=None,recon_speed=1,shuf_speed=1,rchunk=2,cchunk=2,cycspeed=1,cycaa=False):
#     srcdst=arm_base_module_set(srcdst,prefix,maxT,capacity,feat_ch,tr_meta_path,expf,fecnt+1,wemb,wrej,inplace,detached_ta=detached_ta,detached_ga_proto=detached_ga_proto,dropf=dropf,dropp=dropp);
#     srcdst[prefix+"p_recon"]=config_dcganN_insnorm(feat_ch);
#     srcdst=arm_protorec_v2S(srcdst,prefix,feat_ch,speed=recon_speed);
#     srcdst=arm_v2_shufDPS(srcdst,prefix,feat_ch,speed=shuf_speed,rchunk=rchunk,cchunk=cchunk);
#     srcdst=arm_cyc_fe(srcdst,prefix,feat_ch,"res4",feat_ch,aa=cycaa);
#     srcdst=arm_cyc_loss(srcdst,prefix,wcls=cycspeed);
#     return srcdst;
#
#
# def arm_v2IN_prec_shuf_module_setDPS_cyc3o(srcdst,prefix,maxT,capacity,feat_ch,tr_meta_path,expf=1,fecnt=3,wemb=0.3,wrej=0.1,inplace=True,detached_ta=False,detached_ga_proto=False,dropf=0,dropp=None,recon_speed=1,shuf_speed=1,rchunk=2,cchunk=2,cycspeed=1):
#     srcdst=arm_base_module_set(srcdst,prefix,maxT,capacity,feat_ch,tr_meta_path,expf,fecnt+1,wemb,wrej,inplace,detached_ta=detached_ta,detached_ga_proto=detached_ga_proto,dropf=dropf,dropp=dropp);
#     srcdst[prefix+"p_recon"]=config_dcganN_insnorm(feat_ch);
#     srcdst=arm_protorec_v2S(srcdst,prefix,feat_ch,speed=recon_speed);
#     srcdst=arm_v2_shufDPS(srcdst,prefix,feat_ch,speed=shuf_speed,rchunk=rchunk,cchunk=cchunk);
#     srcdst=arm_cyc_fe(srcdst,prefix,feat_ch,"res4",feat_ch);
#     srcdst=arm_cyc_loss_o(srcdst,prefix,wcls=cycspeed,too_dirty_frac=0.2);
#     return srcdst;
# def arm_v2IN_prec_shufA_module_setDPS_cyc3o(srcdst,prefix,maxT,capacity,feat_ch,tr_meta_path,expf=1,fecnt=3,wemb=0.3,wrej=0.1,inplace=True,detached_ta=False,detached_ga_proto=False,dropf=0,dropp=None,recon_speed=1,shuf_speed=1,rchunk=2,cchunk=2,cycspeed=1):
#     srcdst=arm_base_module_set(srcdst,prefix,maxT,capacity,feat_ch,tr_meta_path,expf,fecnt+1,wemb,wrej,inplace,detached_ta=detached_ta,detached_ga_proto=detached_ga_proto,dropf=dropf,dropp=dropp);
#     srcdst[prefix+"p_recon"]=config_dcganN_insnorm(feat_ch);
#     srcdst=arm_protorec_v2S(srcdst,prefix,feat_ch,speed=recon_speed);
#     srcdst=arm_v2_shufADPS(srcdst,prefix,feat_ch,speed=shuf_speed,rchunk=rchunk,cchunk=cchunk);
#     srcdst=arm_cyc_fe(srcdst,prefix,feat_ch,"res4",feat_ch);
#     srcdst=arm_cyc_loss_o(srcdst,prefix,wcls=cycspeed,too_dirty_frac=0.2);
#     return srcdst;
#     srcdst[prefix + "p_recon"] = config_dcganN_insnorm(feat_ch);
#     srcdst = arm_protorec_v2S(srcdst, prefix, feat_ch, speed=recon_speed);
#     srcdst = arm_v2_shufDPS(srcdst, prefix, feat_ch, speed=shuf_speed, rchunk=rchunk, cchunk=cchunk);
#     srcdst=arm_cyc_fe(srcdst,prefix,feat_ch,"res4",feat_ch,aa=cycaa);
#     srcdst=arm_cyc_loss(srcdst,prefix,wcls=cycspeed);
#     return srcdst;
# def arm_v2INXL_prec_shuf_module_setDPS_cyc3o(srcdst,prefix,maxT,capacity,feat_ch,tr_meta_path,expf=1.5,fecnt=3,wemb=0.3,wrej=0.1,inplace=True,detached_ta=False,detached_ga_proto=False,dropf=0,dropp=None,recon_speed=1,shuf_speed=1,rchunk=2,cchunk=2,cycspeed=1):
#     srcdst = arm_base_module_setXL(srcdst, prefix, maxT, capacity, feat_ch, tr_meta_path, expf, fecnt+1, wemb, wrej,
#                                    inplace, detached_ta=detached_ta, detached_ga_proto=detached_ga_proto, dropf=dropf,
#                                    dropp=dropp);
#     srcdst[prefix + "p_recon"] = config_dcganN_insnorm(feat_ch);
#     srcdst = arm_protorec_v2S(srcdst, prefix, feat_ch, speed=recon_speed);
#     srcdst = arm_v2_shufDPS(srcdst, prefix, feat_ch, speed=shuf_speed, rchunk=rchunk, cchunk=cchunk);
#     srcdst=arm_cyc_fe(srcdst,prefix,feat_ch,"res4",feat_ch);
#     srcdst=arm_cyc_loss_o(srcdst,prefix,wcls=cycspeed);
#     return srcdst;
#
# def arm_v2INXL_prec_shuf_module_setDPS_cyc3IN(srcdst,prefix,maxT,capacity,feat_ch,tr_meta_path,expf=1.5,fecnt=3,wemb=0.3,wrej=0.1,inplace=True,detached_ta=False,detached_ga_proto=False,dropf=0,dropp=None,recon_speed=1,shuf_speed=1,rchunk=2,cchunk=2,cycspeed=1):
#     srcdst = arm_base_module_setXLIN(srcdst, prefix, maxT, capacity, feat_ch, tr_meta_path, expf, [fecnt,1], wemb, wrej,
#                                    inplace, detached_ta=detached_ta, detached_ga_proto=detached_ga_proto, dropf=dropf,
#                                    dropp=dropp);
#     srcdst[prefix + "p_recon"] = config_dcganN_insnorm(feat_ch);
#     srcdst = arm_protorec_v2S(srcdst, prefix, feat_ch, speed=recon_speed);
#     srcdst = arm_v2_shufDPS(srcdst, prefix, feat_ch, speed=shuf_speed, rchunk=rchunk, cchunk=cchunk);
#     srcdst=arm_cyc_fe(srcdst,prefix,feat_ch,"res4",feat_ch);
#     srcdst=arm_cyc_loss(srcdst,prefix,wcls=cycspeed);
#     return srcdst;
#
#
# def arm_v2IN_prec_shuf_module_setDPS_cyc3IN(srcdst,prefix,maxT,capacity,feat_ch,tr_meta_path,expf=1,fecnt=3,wemb=0.3,wrej=0.1,inplace=True,detached_ta=False,detached_ga_proto=False,dropf=0,dropp=None,recon_speed=1,shuf_speed=1,rchunk=2,cchunk=2,cycspeed=1):
#     srcdst=arm_base_module_set(srcdst,prefix,maxT,capacity,feat_ch,tr_meta_path,expf,[fecnt,1],wemb,wrej,inplace,detached_ta=detached_ta,detached_ga_proto=detached_ga_proto,dropf=dropf,dropp=dropp);
#     srcdst[prefix+"p_recon"]=config_dcganN_insnorm(feat_ch);
#     srcdst=arm_protorec_v2S(srcdst,prefix,feat_ch,speed=recon_speed);
#     srcdst=arm_v2_shufDPS(srcdst,prefix,feat_ch,speed=shuf_speed,rchunk=rchunk,cchunk=cchunk);
#     srcdst=arm_cyc_fe(srcdst,prefix,feat_ch,"res4",feat_ch);
#     srcdst=arm_cyc_loss(srcdst,prefix,wcls=cycspeed);
#     return srcdst;
#
# def arm_v2IN_prec_shuf_module_setDPS_cyc3sp2(srcdst,prefix,maxT,capacity,feat_ch,tr_meta_path,expf=1,fecnt=3,wemb=0.3,wrej=0.1,inplace=True,detached_ta=False,detached_ga_proto=False,dropf=0,dropp=None,recon_speed=1,shuf_speed=1,rchunk=2,cchunk=2,cycspeed=1):
#     srcdst=arm_base_module_set(srcdst,prefix,maxT,capacity,feat_ch,tr_meta_path,expf,fecnt+1,wemb,wrej,inplace,detached_ta=detached_ta,detached_ga_proto=detached_ga_proto,dropf=dropf,dropp=dropp);
#     srcdst[prefix+"p_recon"]=config_dcganN_insnorm(feat_ch);
#     srcdst=arm_protorec_v2S(srcdst,prefix,feat_ch,speed=recon_speed);
#     srcdst=arm_v2_shufDPS(srcdst,prefix,feat_ch,speed=shuf_speed,rchunk=rchunk,cchunk=cchunk);
#     srcdst=arm_cyc_fe(srcdst,prefix,feat_ch,"res2",feat_ch);
#     srcdst=arm_cyc_loss(srcdst,prefix,wcls=cycspeed);
#     return srcdst;
#
# def arm_v2IN_prec_shuf_module_setDPS_cyc3sp3(srcdst,prefix,maxT,capacity,feat_ch,tr_meta_path,expf=1,fecnt=3,wemb=0.3,wrej=0.1,inplace=True,detached_ta=False,detached_ga_proto=False,dropf=0,dropp=None,recon_speed=1,shuf_speed=1,rchunk=2,cchunk=2,cycspeed=1):
#     srcdst=arm_base_module_set(srcdst,prefix,maxT,capacity,feat_ch,tr_meta_path,expf,fecnt+1,wemb,wrej,inplace,detached_ta=detached_ta,detached_ga_proto=detached_ga_proto,dropf=dropf,dropp=dropp);
#     srcdst[prefix+"p_recon"]=config_dcganN_insnorm(feat_ch);
#     srcdst=arm_protorec_v2S(srcdst,prefix,feat_ch,speed=recon_speed);
#     srcdst=arm_v2_shufDPS(srcdst,prefix,feat_ch,speed=shuf_speed,rchunk=rchunk,cchunk=cchunk);
#     srcdst=arm_cyc_fe(srcdst,prefix,feat_ch,"res3",feat_ch);
#     srcdst=arm_cyc_loss(srcdst,prefix,wcls=cycspeed);
#     return srcdst;