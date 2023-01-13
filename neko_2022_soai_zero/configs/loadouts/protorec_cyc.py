from neko_2022_soai_zero.configs.modules.config_ch2fe import config_ch2fe
from neko_2021_mjt.configs.bogo_modules.config_res_binorm import config_bogo_resbinorm
from neko_2021_mjt.configs.modules.config_sa import config_sa
from neko_2021_mjt.configs.modules.config_ospred import config_linxcs
from neko_2021_mjt.configs.modules.config_cls_emb_loss import config_cls_loss

def arm_cyc_fe(srcdst,prefix,detach_ga,key,feat_ch,aa=False):
    srcdst[prefix+"recon_fe_bbn"]= config_bogo_resbinorm(prefix + "feature_extractor_container", key);
    if(aa):
        srcdst[prefix + "GA_cyc"] = config_sa(feat_ch=feat_ch);
        srcdst[prefix + "recon_char_fe"] = config_ch2fe(prefix + "recon_fe_bbn", prefix + "GA_cyc", detached_ga=detach_ga);
    else:
        srcdst[prefix+"recon_char_fe"]=config_ch2fe(prefix+"recon_fe_bbn",prefix+"GA",detached_ga=detach_ga);
    srcdst[prefix+"recon_char_pred"]= config_linxcs();
    return srcdst;

def arm_cyc_loss(srcdst,prefix,wcls=1):
    srcdst[prefix + "f_recon_loss"] = config_cls_loss(wcls=wcls);
    return srcdst;

#
# def arm_cyc(srcdst,prefix,feat_ch,detach_ga=False):
#     srcdst[prefix+"f_recon"]=config_dcgan(feat_ch,32);
#     srcdst[prefix+"recon_fe_bbn"]= config_bogo_resbinorm(prefix + "feature_extractor_container", "res3");
#     srcdst[prefix+"recon_char_fe"]=config_ch2fe(prefix+"recon_fe_bbn",prefix+"GA",detached_ga=detach_ga);
#     srcdst[prefix+"recon_char_pred"]= config_linxos();
#     srcdst[prefix + "f_recon_loss"] = config_cls_emb_lossohem();
#     return srcdst;
#
#
# # def arm_cycN(srcdst,prefix,feat_ch):
# #     srcdst[prefix+"f_recon"]=config_dcganN(feat_ch,32);
# #     srcdst[prefix+"recon_fe_bbn"]= config_bogo_resbinorm(prefix + "feature_extractor_container", "res3");
# #     srcdst[prefix+"recon_char_fe"]=config_ch2fe(prefix+"recon_fe_bbn",prefix+"GA");
# #     srcdst[prefix+"recon_char_pred"]= config_linxos();
# #     srcdst[prefix + "f_recon_loss"] = config_cls_emb_loss2(0,0.1);
# #     return srcdst;
# #
#
#
# def arm_protorecg2(srcdst,prefix,feat_ch):
#     srcdst[prefix+"p_recon"]=config_dcgan(feat_ch);
#     srcdst[prefix+"p_recon_loss"]=config_masked_l2();
#     return srcdst;
#
#
# def arm_protorecg2N(srcdst,prefix,feat_ch):
#     srcdst[prefix+"p_recon"]=config_dcganN(feat_ch);
#     srcdst[prefix+"p_recon_loss"]=config_masked_l2();
#     return srcdst;
#
# def arm_rec_cyc_module_set(srcdst,prefix,maxT,capacity,feat_ch,tr_meta_path,expf=1,fecnt=3,wemb=0.3,wrej=0.1,inplace=True):
#     srcdst=arm_base_module_set(srcdst,prefix,maxT,capacity,feat_ch,tr_meta_path,expf,fecnt,wemb,wrej,inplace);
#     srcdst=arm_protorecg2(srcdst,prefix,feat_ch);
#     srcdst=arm_cyc(srcdst,prefix,feat_ch)
#     return srcdst;
#
#
#
#
# def arm_rec_cycN_module_set(srcdst,prefix,maxT,capacity,feat_ch,tr_meta_path,expf=1,fecnt=3,wemb=0.3,wrej=0.1,inplace=True):
#     srcdst=arm_base_module_set(srcdst,prefix,maxT,capacity,feat_ch,tr_meta_path,expf,fecnt,wemb,wrej,inplace);
#     srcdst=arm_protorecg2N(srcdst,prefix,feat_ch);
#     srcdst=arm_cycN(srcdst,prefix,feat_ch)
#     return srcdst;
#
#
#
#
# # share 01, shares the p recon and the F recon
# def arm_cycNS(srcdst,prefix,feat_ch,detach_ga):
#     srcdst[prefix+"f_recon"]=config_softlink(prefix+"p_recon");
#     srcdst[prefix+"recon_fe_bbn"]= config_bogo_resbinorm(prefix + "feature_extractor_container", "res3");
#     srcdst[prefix+"recon_char_fe"]=config_ch2fe(prefix+"recon_fe_bbn",prefix+"GA",detached_ga=detach_ga);
#     srcdst[prefix+"recon_char_pred"]= config_linxos();
#     srcdst[prefix + "f_recon_loss"] = config_cls_emb_loss2(0,0.1);
#     return srcdst;
#
# def arm_cycNsp(srcdst,prefix,feat_ch,detach_ga):
#     srcdst[prefix+"f_recon"]=config_dcganN(feat_ch,32);
#     srcdst[prefix + "recon_fe_bbn"] = config_bogo_resbinorm(prefix + "feature_extractor_container", "res2");
#     srcdst[prefix + "recon_char_fe"] = config_ch2fe(prefix + "recon_fe_bbn", prefix + "GA", detached_ga=detach_ga);
#     srcdst[prefix + "recon_char_pred"] = config_linxos();
#     srcdst[prefix + "f_recon_loss"] = config_cls_emb_loss2(0, 0.1);
#     return srcdst;
#
# def arm_cycN(srcdst,prefix,feat_ch,detach_ga):
#     srcdst[prefix+"f_recon"]=config_dcganN(feat_ch,32);
#     srcdst[prefix + "recon_fe_bbn"] = config_bogo_resbinorm(prefix + "feature_extractor_container", "res3");
#     srcdst[prefix + "recon_char_fe"] = config_ch2fe(prefix + "recon_fe_bbn", prefix + "GA", detached_ga=detach_ga);
#     srcdst[prefix + "recon_char_pred"] = config_linxcs();
#     srcdst[prefix + "f_recon_loss"] = config_cls_emb_loss2(0, 0.1);
#     return srcdst;
# def arm_cyc_fe(srcdst,prefix,detach_ga,key,feat_ch,aa=False):
#     srcdst[prefix+"recon_fe_bbn"]= config_bogo_resbinorm(prefix + "feature_extractor_container", key);
#     if(aa):
#         srcdst[prefix + "GA_cyc"] = config_sa(feat_ch=feat_ch);
#         srcdst[prefix + "recon_char_fe"] = config_ch2fe(prefix + "recon_fe_bbn", prefix + "GA_cyc", detached_ga=detach_ga);
#     else:
#         srcdst[prefix+"recon_char_fe"]=config_ch2fe(prefix+"recon_fe_bbn",prefix+"GA",detached_ga=detach_ga);
#     srcdst[prefix+"recon_char_pred"]= config_linxcs();
#     return srcdst;
# def arm_cyc_loss(srcdst,prefix,wcls=1):
#     srcdst[prefix + "f_recon_loss"] = config_cls_loss(wcls=wcls);
#     return srcdst;
#
# def arm_cyc_loss_o(srcdst,prefix,wcls=1,too_dirty_frac=0.3):
#     srcdst[prefix + "f_recon_loss"] = config_cls_emb_lossosd(wcls=wcls,wemb=0,dirty_frac=too_dirty_frac,too_simple_frac=0.2);
#     return srcdst;
# # share 02 shares the p recon and the F recon, and protofe and reconfe
# def arm_cycNSsp(srcdst,prefix,feat_ch,detach_ga):
#     srcdst[prefix+"f_recon"]=config_softlink(prefix+"p_recon");
#     srcdst=arm_cyc_fe(srcdst,prefix,detach_ga,"res2",feat_ch)
#
#     srcdst[prefix + "f_recon_loss"] = config_cls_emb_loss2(0,0.1);
#     return srcdst;
# def arm_mvap(srcdst,prefix,feat_ch,detach_ga):
#     srcdst["recon_mva"]=config_linxos()
#     return srcdst;
# def arm_rec_cycNS_module_set(srcdst,prefix,maxT,capacity,feat_ch,tr_meta_path,expf=1,fecnt=3,wemb=0.3,wrej=0.1,inplace=True,detach_TA=False,detach_GA_proto=False):
#     srcdst=arm_base_module_set(srcdst,prefix,maxT,capacity,feat_ch,tr_meta_path,expf,fecnt,wemb,wrej,inplace,detached_ta=detach_TA,detached_ga_proto=detach_GA_proto);
#     srcdst=arm_protorecg2N(srcdst,prefix,feat_ch);
#     srcdst=arm_cycNS(srcdst,prefix,feat_ch,detach_ga=detach_GA_proto)
#     return srcdst;
#
# def arm_rec_cycNSsp_module_set(srcdst,prefix,maxT,capacity,feat_ch,tr_meta_path,expf=1,fecnt=3,wemb=0.3,wrej=0.1,inplace=True,detach_TA=False,detach_GA_proto=False):
#     srcdst=arm_base_module_set(srcdst,prefix,maxT,capacity,feat_ch,tr_meta_path,expf,fecnt,wemb,wrej,inplace,detached_ta=detach_TA,detached_ga_proto=detach_GA_proto);
#     srcdst=arm_protorecg2N(srcdst,prefix,feat_ch);
#     srcdst=arm_cycNSsp(srcdst,prefix,feat_ch,detach_ga=detach_GA_proto)
#     return srcdst;
# def arm_rec_cycNSsp_mvap_module_set(srcdst,prefix,maxT,capacity,feat_ch,tr_meta_path,expf=1,fecnt=3,wemb=0.3,wrej=0.1,inplace=True,detach_TA=False,detach_GA_proto=False):
#     srcdst=arm_base_module_set(srcdst,prefix,maxT,capacity,feat_ch,tr_meta_path,expf,fecnt,wemb,wrej,inplace,detached_ta=detach_TA,detached_ga_proto=detach_GA_proto);
#     srcdst=arm_protorecg2N(srcdst,prefix,feat_ch);
#     srcdst=arm_cycNSsp(srcdst,prefix,feat_ch,detach_ga=detach_GA_proto)
#     return srcdst;
#
#
# def arm_dommixMVA_rec_cycNS_module_set(srcdst,prefix,maxT,capacity,feat_ch,tr_meta_path,expf=1,fecnt=3,wemb=0.3,wrej=0.1,inplace=True,detach_TA=False,detach_GA_proto=False):
#     srcdst=arm_base_module_set(srcdst,prefix,maxT,capacity,feat_ch,tr_meta_path,expf,fecnt,wemb,wrej,inplace,detached_ta=detach_TA,detached_ga_proto=detach_GA_proto);
#     srcdst=arm_protorecg2N(srcdst,prefix,feat_ch);
#     srcdst=arm_cycNS(srcdst,prefix,feat_ch,detach_GA_proto);
#     srcdst=arm_dom_mix_mva(srcdst,prefix);
#     return srcdst;
#
# from neko_2022_soai_zero.configs.loadouts.domd import arm_dom_mix_slacky;
#
# def arm_dommixFS_rec_cycNS_module_set(srcdst,prefix,maxT,capacity,feat_ch,tr_meta_path,expf=1,fecnt=3,wemb=0.3,wrej=0.1,inplace=True,detach_TA=False,detach_GA_proto=False):
#     srcdst=arm_base_module_set(srcdst,prefix,maxT,capacity,feat_ch,tr_meta_path,expf,fecnt,wemb,wrej,inplace,detached_ta=detach_TA,detached_ga_proto=detach_GA_proto);
#     srcdst=arm_protorecg2N(srcdst,prefix,feat_ch);
#     srcdst=arm_cycNS(srcdst,prefix,feat_ch,detach_ga=detach_GA_proto);
#     srcdst=arm_dom_mix_slacky(srcdst,prefix,feat_ch);
#     return srcdst;
