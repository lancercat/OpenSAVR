from neko_2022_soai_zero.routines.ocr_routine.mk7g3.osdan_routine_mk7g3_water2 import neko_HDOS2C_routine_CFmk7g3_rec_cyc3ks,neko_HDOS2C_routine_CFmk7g3_cyc3ksanfr

def osdanmk7g3_rec_cyc3ks_family_ocr_routine(rotine_impl, sampler_name, prototyper_name, feature_extractor_name, seq_name,
                           CAMname, pred_name,ctx_name, loss_name,dom_loss_name,
                                   prec_name,prec_loss_name,fpm_recon_loss_name,
                                    proto_part_recon_name,proto_part_recon_loss_name,
                                    shuf_img_name,shuf_recon_name,shuf_recon_loss_name,shuf_sim_loss_name,shuf_proto_name,
                                    shuf_part_recon_name,shuf_part_recon_loss_name,inv_loss_name,
                                   frec_name,recon_char_fe_name,recon_char_pred_name, frec_loss_name,
                           label_name, image_name, log_path, log_each, name, maxT):
    dic = {
        "maxT": maxT,
        "name": name,
        "routine": rotine_impl,
        "mod_cvt_dicts":
            {
                "sampler": sampler_name,
                "prototyper": prototyper_name,
                "feature_extractor": feature_extractor_name,
                "CAM": CAMname,
                "seq": seq_name,
                "preds": pred_name,
                "losses": loss_name,
                "ctxmod": ctx_name,
                "dom_mix": dom_loss_name,
                "p_recon": prec_name,
                "p_recon_loss": prec_loss_name,
                "proto_part_recon": proto_part_recon_name,
                "proto_part_recon_loss": proto_part_recon_loss_name,
                "shuf_part_recon": shuf_part_recon_name,
                "shuf_part_recon_loss": shuf_part_recon_loss_name,
                "fpm_recon_loss": fpm_recon_loss_name,
                "inv_loss":inv_loss_name,
                "shuf_img": shuf_img_name,
                "shuf_recon": shuf_recon_name,
                "shuf_recon_loss": shuf_recon_loss_name,
                "shuf_sim_loss": shuf_sim_loss_name,
                "shuf_proto": shuf_proto_name,
                "f_recon": frec_name,
                "recon_char_fe": recon_char_fe_name,
                "recon_char_pred": recon_char_pred_name,
                "f_recon_loss": frec_loss_name,
            },
        "inp_cvt_dicts":
            {
                "label": label_name,
                "image": image_name,
            },
        "log_path": log_path,
        "log_each": log_each,
    }
    return dic;

def osdanmk7g3_rec_cyc3ks_ocr_routine(**kwargs):
    return osdanmk7g3_rec_cyc3ks_family_ocr_routine(neko_HDOS2C_routine_CFmk7g3_rec_cyc3ks,**kwargs)
def osdanmk7g3_rec_cyc3ksanfr_ocr_routine(**kwargs):
    return osdanmk7g3_rec_cyc3ks_family_ocr_routine(neko_HDOS2C_routine_CFmk7g3_cyc3ksanfr,**kwargs)

#
# from neko_2022_soai_zero.routines.ocr_routine.mk7g3.osdan_routine_mk7g3_water2 import \
#     neko_HDOS2C_routine_CFmk7g3_rec_cyc,\
#     neko_HDOS2C_routine_CFmk7g3_rec_cyc2,\
#     neko_HDOS2C_routine_CFmk7g3_rec_cyc3,neko_HDOS2C_routine_CFmk7g3_cyc3k,\
#     neko_HDOS2C_routine_CFmk7g3_rec_cyc3k,neko_HDOS2C_routine_CFmk7g3_rec_cyc3k128,neko_HDOS2C_routine_CFmk7g3_rec,\
#     neko_HDOS2C_routine_CFmk7g3_cyc3k_dmxs,neko_HDOS2C_routine_CFmk7g3_rec_cyc3ks,neko_HDOS2C_routine_CFmk7g3R_rec_cyc3ks,neko_HDOS2C_routine_CFmk7g3_rec_cyc3ksa,\
#     neko_HDOS2C_routine_CFmk7g3_rec_cyc3ks128,neko_HDOS2C_routine_CFmk7g3_rec_cyc3ksa128,\
#     neko_HDOS2C_routine_CFmk7g3_rec_cyc3kss,neko_HDOS2C_routine_CFmk7g3_rec_cyc3ksas,\
#     neko_HDOS2C_routine_CFmk7g3_rec_cyc3ksasd,neko_HDOS2C_routine_CFmk7g3_rec_cyc3ksasbd,\
#     neko_HDOS2C_routine_CFmk7g3_rec_cyc3ksasbd_cycrfp,neko_HDOS2C_routine_CFmk7g3_rec_cyc3ksasbd_cycrf,\
# neko_HDOS2C_routine_CFmk7g3_rec_cyc3ksasbd_cycrp,neko_HDOS2C_routine_CFmk7g3_rec_cyc3ksasbd_cycrx,\
#     neko_HDOS2C_routine_CFmk7g3_rec_cyc3ksasbd_cycrfpnd,neko_HDOS2C_routine_CFmk7g3_rec_cyc3ksasbd_cycrfnd,\
#     neko_HDOS2C_routine_CFmk7g3_rec_cyc3ksasbd_cycrftfr,neko_HDOS2C_routine_CFmk7g3_rec_cyc3ksasbd_cycrfndtfr,\
# neko_HDOS2C_routine_CFmk7g3_rec_cyc3ksasbds,neko_HDOS2C_routine_CFmk7g3_rec_cyc3ksg,neko_HDOS2C_routine_CFmk7g3_rec_cyc3ksasbs,\
# neko_HDOS2C_routine_CFmk7g3_rec_cyc3ksak,neko_HDOS2C_routine_CFmk7g3_rec_cyc3ksakd,neko_HDOS2C_routine_CFmk7g3_rec_cyc3ksdg,neko_HDOS2C_routine_CFmk7g3_rec_cyc3ksad,neko_HDOS2C_routine_CFmk7g3_cyc3ks_dmxs,\
#     neko_HDOS2C_routine_CFmk7g3_cyc3ksnfr,neko_HDOS2C_routine_CFmk7g3_cyc3ksnfrd,\
#     neko_HDOS2C_routine_CFmk7g3_cyc3ksanfr,neko_HDOS2C_routine_CFmk7g3_cyc3ksgnfr,neko_HDOS2C_routine_CFmk7g3_cyc3ksagnfr,\
# neko_HDOS2C_routine_CFmk7g3_cyc3ks_dmxdp
# from neko_2022_soai_zero.routines.ocr_routine.mk7g3.osdan_routine_mk7g4_water2 import neko_HDOS2C_routine_CFmk7g3_cyc3ksu_dmxs,neko_HDOS2C_routine_CFmk7g3_rec_cyc3ksu
#
#
#
#
#
# def osdanmk7g3_rec_cyc_ocr_routine(sampler_name, prototyper_name, feature_extractor_name, seq_name,
#                            CAMname, pred_name,ctx_name, loss_name,dom_loss_name,
#                                    prec_name,prec_loss_name,
#                                    frec_name,recon_char_fe_name,recon_char_pred_name, frec_loss_name,
#                            label_name, image_name, log_path, log_each, name, maxT):
#     dic={
#         "maxT": maxT,
#         "name":name,
#         "routine":neko_HDOS2C_routine_CFmk7g3_rec_cyc,
#         "mod_cvt_dicts":
#         {
#             "sampler": sampler_name,
#             "prototyper":prototyper_name,
#             "feature_extractor":feature_extractor_name,
#             "CAM":CAMname,
#             "seq": seq_name,
#             "preds":pred_name,
#             "losses":loss_name,
#             "ctxmod": ctx_name,
#             "dom_mix":dom_loss_name,
#             "p_recon": prec_name,
#             "p_recon_loss":prec_loss_name,
#             "f_recon":frec_name,
#             "recon_char_fe": recon_char_fe_name,
#             "recon_char_pred":recon_char_pred_name,
#             "f_recon_loss": frec_loss_name,
#         },
#         "inp_cvt_dicts":
#         {
#             "label":label_name,
#             "image":image_name,
#         },
#         "log_path":log_path,
#         "log_each":log_each,
#     }
#     return dic;
#
#
#
#
# def osdanmk7g3_rec_cyc2_ocr_routine(sampler_name, prototyper_name, feature_extractor_name, seq_name,
#                            CAMname, pred_name,ctx_name, loss_name,dom_loss_name,
#                                    prec_name,prec_loss_name,
#                                    shuf_img_name,shuf_recon_name,shuf_recon_loss_name,shuf_sim_loss,shuf_proto,
#                                    frec_name,recon_char_fe_name,recon_char_pred_name, frec_loss_name,
#                            label_name, image_name, log_path, log_each, name, maxT):
#
#     dic={
#         "maxT": maxT,
#         "name":name,
#         "routine":neko_HDOS2C_routine_CFmk7g3_rec_cyc2,
#         "mod_cvt_dicts":
#         {
#             "sampler": sampler_name,
#             "prototyper":prototyper_name,
#             "feature_extractor":feature_extractor_name,
#             "CAM":CAMname,
#             "seq": seq_name,
#             "preds":pred_name,
#             "losses":loss_name,
#             "ctxmod": ctx_name,
#             "dom_mix":dom_loss_name,
#             "shuf_img":shuf_img_name,
#             "shuf_recon":shuf_recon_name,
#             "shuf_recon_loss":shuf_recon_loss_name,
#             "shuf_sim_loss":shuf_sim_loss,
#             "shuf_proto":shuf_proto,
#             "p_recon": prec_name,
#             "p_recon_loss":prec_loss_name,
#             "f_recon":frec_name,
#             "recon_char_fe": recon_char_fe_name,
#             "recon_char_pred":recon_char_pred_name,
#             "f_recon_loss": frec_loss_name,
#         },
#         "inp_cvt_dicts":
#         {
#             "label":label_name,
#             "image":image_name,
#         },
#         "log_path":log_path,
#         "log_each":log_each,
#     }
#     return dic;
#
#
# def osdanmk7g3_rec_cyc3_ocr_routine(sampler_name, prototyper_name, feature_extractor_name, seq_name,
#                            CAMname, pred_name,ctx_name, loss_name,dom_loss_name,
#                                    prec_name,prec_loss_name,
#                                     shuf_img_name,shuf_recon_name,shuf_recon_loss_name,shuf_sim_loss_name,
#                                    frec_name,recon_char_fe_name,recon_char_pred_name, frec_loss_name,
#                            label_name, image_name, log_path, log_each, name, maxT):
#
#     dic={
#         "maxT": maxT,
#         "name":name,
#         "routine":neko_HDOS2C_routine_CFmk7g3_rec_cyc3,
#         "mod_cvt_dicts":
#         {
#             "sampler": sampler_name,
#             "prototyper":prototyper_name,
#             "feature_extractor":feature_extractor_name,
#             "CAM":CAMname,
#             "seq": seq_name,
#             "preds":pred_name,
#             "losses":loss_name,
#             "ctxmod": ctx_name,
#             "dom_mix":dom_loss_name,
#             "p_recon": prec_name,
#             "p_recon_loss":prec_loss_name,
#             "shuf_img": shuf_img_name,
#             "shuf_recon": shuf_recon_name,
#             "shuf_recon_loss": shuf_recon_loss_name,
#             "shuf_sim_loss":shuf_sim_loss_name,
#             "f_recon":frec_name,
#             "recon_char_fe": recon_char_fe_name,
#             "recon_char_pred":recon_char_pred_name,
#             "f_recon_loss": frec_loss_name,
#         },
#         "inp_cvt_dicts":
#         {
#             "label":label_name,
#             "image":image_name,
#         },
#         "log_path":log_path,
#         "log_each":log_each,
#     }
#     return dic;
#
#
#
#
# def osdanmk7g3_cyc3k_ocr_routine(sampler_name, prototyper_name, feature_extractor_name, seq_name,
#                            CAMname, pred_name,ctx_name, loss_name,dom_loss_name,
#                                    prec_name,prec_loss_name,fpm_recon_loss_name,
#                                     proto_part_recon_name,proto_part_recon_loss_name,
#                                     shuf_img_name,shuf_recon_name,shuf_recon_loss_name,shuf_sim_loss_name,shuf_proto_name,
#                                     shuf_part_recon_name,shuf_part_recon_loss_name,
#                                    frec_name,recon_char_fe_name,recon_char_pred_name, frec_loss_name,
#                            label_name, image_name, log_path, log_each, name, maxT):
#
#     dic={
#         "maxT": maxT,
#         "name":name,
#         "routine":neko_HDOS2C_routine_CFmk7g3_cyc3k,
#         "mod_cvt_dicts":
#         {
#             "sampler": sampler_name,
#             "prototyper":prototyper_name,
#             "feature_extractor":feature_extractor_name,
#             "CAM":CAMname,
#             "seq": seq_name,
#             "preds":pred_name,
#             "losses":loss_name,
#             "ctxmod": ctx_name,
#             "dom_mix":dom_loss_name,
#             "p_recon": prec_name,
#             "p_recon_loss":prec_loss_name,
#             "proto_part_recon": proto_part_recon_name,
#             "proto_part_recon_loss": proto_part_recon_loss_name,
#             "shuf_part_recon": shuf_part_recon_name,
#             "shuf_part_recon_loss": shuf_part_recon_loss_name,
#             "shuf_proto":shuf_proto_name,
#             "shuf_img": shuf_img_name,
#             "shuf_recon": shuf_recon_name,
#             "shuf_recon_loss": shuf_recon_loss_name,
#             "shuf_sim_loss":shuf_sim_loss_name,
#             "f_recon":frec_name,
#             "recon_char_fe": recon_char_fe_name,
#             "recon_char_pred":recon_char_pred_name,
#             "f_recon_loss": frec_loss_name,
#             "fpm_recon_loss": fpm_recon_loss_name
#         },
#         "inp_cvt_dicts":
#         {
#             "label":label_name,
#             "image":image_name,
#         },
#         "log_path":log_path,
#         "log_each":log_each,
#     }
#     return dic;
#
#
# def osdanmk7g3_cyc3k_mvas_ocr_routine(sampler_name, prototyper_name, feature_extractor_name, seq_name,
#                            CAMname, pred_name,ctx_name, loss_name,dom_loss_name,
#                                    prec_name,prec_loss_name,
#                                     proto_part_recon_name,proto_part_recon_loss_name,
#                                     shuf_img_name,shuf_recon_name,shuf_recon_loss_name,shuf_sim_loss_name,shuf_proto_name,
#                                     shuf_part_recon_name,shuf_part_recon_loss_name,
#                                    frec_name,recon_char_fe_name,recon_char_pred_name, frec_loss_name,
#                            label_name, image_name, log_path, log_each, name, maxT):
#
#     dic={
#         "maxT": maxT,
#         "name":name,
#         "routine":neko_HDOS2C_routine_CFmk7g3_cyc3k_dmxs,
#         "mod_cvt_dicts":
#         {
#             "sampler": sampler_name,
#             "prototyper":prototyper_name,
#             "feature_extractor":feature_extractor_name,
#             "CAM":CAMname,
#             "seq": seq_name,
#             "preds":pred_name,
#             "losses":loss_name,
#             "ctxmod": ctx_name,
#             "dom_mix":dom_loss_name,
#             "p_recon": prec_name,
#             "p_recon_loss":prec_loss_name,
#             "proto_part_recon": proto_part_recon_name,
#             "proto_part_recon_loss": proto_part_recon_loss_name,
#             "shuf_part_recon": shuf_part_recon_name,
#             "shuf_part_recon_loss": shuf_part_recon_loss_name,
#             "shuf_img": shuf_img_name,
#             "shuf_recon": shuf_recon_name,
#             "shuf_recon_loss": shuf_recon_loss_name,
#             "shuf_sim_loss":shuf_sim_loss_name,
#             "shuf_proto":shuf_proto_name,
#             "f_recon":frec_name,
#             "recon_char_fe": recon_char_fe_name,
#             "recon_char_pred":recon_char_pred_name,
#             "f_recon_loss": frec_loss_name,
#         },
#         "inp_cvt_dicts":
#         {
#             "label":label_name,
#             "image":image_name,
#         },
#         "log_path":log_path,
#         "log_each":log_each,
#     }
#     return dic;
#
# def osdanmk7g3_rec_cyc3k_ocr_routine(sampler_name, prototyper_name, feature_extractor_name, seq_name,
#                            CAMname, pred_name,ctx_name, loss_name,dom_loss_name,
#                                    prec_name,prec_loss_name,fpm_recon_loss_name,
#                                     proto_part_recon_name,proto_part_recon_loss_name,
#                                     shuf_img_name,shuf_recon_name,shuf_recon_loss_name,shuf_sim_loss_name,shuf_proto_name,
#                                     shuf_part_recon_name,shuf_part_recon_loss_name,
#                                    frec_name,recon_char_fe_name,recon_char_pred_name, frec_loss_name,
#                            label_name, image_name, log_path, log_each, name, maxT):
#
#     dic={
#         "maxT": maxT,
#         "name":name,
#         "routine":neko_HDOS2C_routine_CFmk7g3_rec_cyc3k,
#         "mod_cvt_dicts":
#         {
#             "sampler": sampler_name,
#             "prototyper": prototyper_name,
#             "feature_extractor": feature_extractor_name,
#             "CAM": CAMname,
#             "seq": seq_name,
#             "preds": pred_name,
#             "losses": loss_name,
#             "ctxmod": ctx_name,
#             "dom_mix": dom_loss_name,
#             "p_recon": prec_name,
#             "p_recon_loss": prec_loss_name,
#             "proto_part_recon": proto_part_recon_name,
#             "proto_part_recon_loss": proto_part_recon_loss_name,
#             "shuf_part_recon": shuf_part_recon_name,
#             "shuf_part_recon_loss": shuf_part_recon_loss_name,
#             "shuf_img": shuf_img_name,
#             "shuf_recon": shuf_recon_name,
#             "shuf_recon_loss": shuf_recon_loss_name,
#             "shuf_sim_loss": shuf_sim_loss_name,
#             "shuf_proto":shuf_proto_name,
#             "f_recon": frec_name,
#             "recon_char_fe": recon_char_fe_name,
#             "recon_char_pred": recon_char_pred_name,
#             "f_recon_loss": frec_loss_name,
#             "fpm_recon_loss":fpm_recon_loss_name
#         },
#         "inp_cvt_dicts":
#         {
#             "label":label_name,
#             "image":image_name,
#         },
#         "log_path":log_path,
#         "log_each":log_each,
#     }
#     return dic;
#
#
# def osdanmk7g3_rec_ocr_routine(sampler_name, prototyper_name, feature_extractor_name, seq_name,
#                            CAMname, pred_name,ctx_name, loss_name,dom_loss_name,
#                                    prec_name,prec_loss_name,
#                                     shuf_img_name,shuf_recon_name,shuf_recon_loss_name,shuf_sim_loss_name,shuf_proto_name,
#                                    frec_name,recon_char_fe_name,recon_char_pred_name, frec_loss_name,
#                            label_name, image_name, log_path, log_each, name, maxT):
#
#     dic={
#         "maxT": maxT,
#         "name":name,
#         "routine":neko_HDOS2C_routine_CFmk7g3_rec,
#         "mod_cvt_dicts":
#         {
#             "sampler": sampler_name,
#             "prototyper":prototyper_name,
#             "feature_extractor":feature_extractor_name,
#             "CAM":CAMname,
#             "seq": seq_name,
#             "preds":pred_name,
#             "losses":loss_name,
#             "ctxmod": ctx_name,
#             "dom_mix":dom_loss_name,
#             "p_recon": prec_name,
#             "p_recon_loss":prec_loss_name,
#             "shuf_img": shuf_img_name,
#             "shuf_recon": shuf_recon_name,
#             "shuf_recon_loss": shuf_recon_loss_name,
#             "shuf_sim_loss":shuf_sim_loss_name,
#             "shuf_proto":shuf_proto_name,
#             "f_recon":frec_name,
#             "recon_char_fe": recon_char_fe_name,
#             "recon_char_pred":recon_char_pred_name,
#             "f_recon_loss": frec_loss_name,
#         },
#         "inp_cvt_dicts":
#         {
#             "label":label_name,
#             "image":image_name,
#         },
#         "log_path":log_path,
#         "log_each":log_each,
#     }
#     return dic;
#
#
#
# def osdanmk7g3_rec_cyc3ksnfr_ocr_routine(**kwargs):
#     return osdanmk7g3_rec_cyc3ks_family_ocr_routine(neko_HDOS2C_routine_CFmk7g3_cyc3ksnfr,**kwargs)
#
# def osdanmk7g3_rec_cyc3ksanfr_ocr_routine(**kwargs):
#     return osdanmk7g3_rec_cyc3ks_family_ocr_routine(neko_HDOS2C_routine_CFmk7g3_cyc3ksanfr,**kwargs)
#
# def osdanmk7g3_rec_cyc3ksgnfr_ocr_routine(**kwargs):
#     return osdanmk7g3_rec_cyc3ks_family_ocr_routine(neko_HDOS2C_routine_CFmk7g3_cyc3ksgnfr,**kwargs)
# def osdanmk7g3_rec_cyc3ksagnfr_ocr_routine(**kwargs):
#     return osdanmk7g3_rec_cyc3ks_family_ocr_routine(neko_HDOS2C_routine_CFmk7g3_cyc3ksagnfr,**kwargs)
#
# def osdanmk7g3_rec_cyc3ksnfrd_ocr_routine(**kwargs):
#     return osdanmk7g3_rec_cyc3ks_family_ocr_routine(neko_HDOS2C_routine_CFmk7g3_cyc3ksnfrd,**kwargs)
#
# def osdanmk7g3_rec_cyc3ksu_ocr_routine(**kwargs):
#     return osdanmk7g3_rec_cyc3ks_family_ocr_routine(neko_HDOS2C_routine_CFmk7g3_rec_cyc3ksu,**kwargs)
#
#
# def osdanmk7g3_rec_cyc3ks_dmxs_ocr_routine(**kwargs):
#     return osdanmk7g3_rec_cyc3ks_family_ocr_routine(neko_HDOS2C_routine_CFmk7g3_cyc3ks_dmxs,**kwargs)
# def osdanmk7g3_rec_cyc3ks_dmxdp_ocr_routine(**kwargs):
#     return osdanmk7g3_rec_cyc3ks_family_ocr_routine(neko_HDOS2C_routine_CFmk7g3_cyc3ks_dmxdp,**kwargs)
#
#
# def osdanmk7g3_rec_cyc3ksg_ocr_routine(**kwargs):
#     return osdanmk7g3_rec_cyc3ks_family_ocr_routine(neko_HDOS2C_routine_CFmk7g3_rec_cyc3ksg,**kwargs)
# def osdanmk7g3_rec_cyc3ksdg_ocr_routine(**kwargs):
#     return osdanmk7g3_rec_cyc3ks_family_ocr_routine(neko_HDOS2C_routine_CFmk7g3_rec_cyc3ksdg,**kwargs)
# def osdanmk7g3_rec_cyc3ks128_ocr_routine(**kwargs):
#     return osdanmk7g3_rec_cyc3ks_family_ocr_routine(neko_HDOS2C_routine_CFmk7g3_rec_cyc3ks128,**kwargs)
# def osdanmk7g3_rec_cyc3ksa_ocr_routine(**kwargs):
#     return osdanmk7g3_rec_cyc3ks_family_ocr_routine(neko_HDOS2C_routine_CFmk7g3_rec_cyc3ksa,**kwargs)
# def osdanmk7g3_rec_cyc3ksak_ocr_routine(**kwargs):
#     return osdanmk7g3_rec_cyc3ks_family_ocr_routine(neko_HDOS2C_routine_CFmk7g3_rec_cyc3ksak,**kwargs)
# def osdanmk7g3_rec_cyc3ksakd_ocr_routine(**kwargs):
#     return osdanmk7g3_rec_cyc3ks_family_ocr_routine(neko_HDOS2C_routine_CFmk7g3_rec_cyc3ksakd,**kwargs)
# def osdanmk7g3_rec_cyc3ksad_ocr_routine(**kwargs):
#     return osdanmk7g3_rec_cyc3ks_family_ocr_routine(neko_HDOS2C_routine_CFmk7g3_rec_cyc3ksad,**kwargs)
#
#
#
#
# # More complete detaching
# def osdanmk7g3_rec_cyc3ksasbd_ocr_routine(**kwargs):
#     return osdanmk7g3_rec_cyc3ks_family_ocr_routine(neko_HDOS2C_routine_CFmk7g3_rec_cyc3ksasbd,**kwargs)
# def osdanmk7g3_rec_cyc3ksasbs_ocr_routine(**kwargs):
#     return osdanmk7g3_rec_cyc3ks_family_ocr_routine(neko_HDOS2C_routine_CFmk7g3_rec_cyc3ksasbs,**kwargs)
# def osdanmk7g3_rec_cyc3ksasbds_ocr_routine(**kwargs):
#     return osdanmk7g3_rec_cyc3ks_family_ocr_routine(neko_HDOS2C_routine_CFmk7g3_rec_cyc3ksasbds,**kwargs)
# def osdanmk7g3_rec_cyc3ksasbdrfp_ocr_routine(**kwargs):
#     return osdanmk7g3_rec_cyc3ks_family_ocr_routine(neko_HDOS2C_routine_CFmk7g3_rec_cyc3ksasbd_cycrfp,**kwargs)
# def osdanmk7g3_rec_cyc3ksasbdrp_ocr_routine(**kwargs):
#     return osdanmk7g3_rec_cyc3ks_family_ocr_routine(neko_HDOS2C_routine_CFmk7g3_rec_cyc3ksasbd_cycrp,**kwargs)
# def osdanmk7g3_rec_cyc3ksasbdrx_ocr_routine(**kwargs):
#     return osdanmk7g3_rec_cyc3ks_family_ocr_routine(neko_HDOS2C_routine_CFmk7g3_rec_cyc3ksasbd_cycrx,**kwargs)
#
# def osdanmk7g3_rec_cyc3ksasbdrf_ocr_routine(**kwargs):
#     return osdanmk7g3_rec_cyc3ks_family_ocr_routine(neko_HDOS2C_routine_CFmk7g3_rec_cyc3ksasbd_cycrf,**kwargs)
# def osdanmk7g3_rec_cyc3ksasbdrfpnd_ocr_routine(**kwargs):
#     return osdanmk7g3_rec_cyc3ks_family_ocr_routine(neko_HDOS2C_routine_CFmk7g3_rec_cyc3ksasbd_cycrfpnd,**kwargs)
# def osdanmk7g3_rec_cyc3ksasbdrfnd_ocr_routine(**kwargs):
#     return osdanmk7g3_rec_cyc3ks_family_ocr_routine(neko_HDOS2C_routine_CFmk7g3_rec_cyc3ksasbd_cycrfnd,**kwargs)
# def osdanmk7g3_rec_cyc3ksa128_ocr_routine(**kwargs):
#     return osdanmk7g3_rec_cyc3ks_family_ocr_routine(neko_HDOS2C_routine_CFmk7g3_rec_cyc3ksa128,**kwargs)
# def osdanmk7g3R_rec_cyc3ks_ocr_routine(**kwargs):
#     return osdanmk7g3_rec_cyc3ks_family_ocr_routine(osdanmk7g3R_rec_cyc3ks_ocr_routine,**kwargs)
#
# def osdanmk7g3_rec_cyc3ksas_ocr_routine(**kwargs):
#     return osdanmk7g3_rec_cyc3ks_family_ocr_routine(neko_HDOS2C_routine_CFmk7g3_rec_cyc3ksas,**kwargs)
#
# def osdanmk7g3_rec_cyc3kss_ocr_routine(**kwargs):
#     return osdanmk7g3_rec_cyc3ks_family_ocr_routine(neko_HDOS2C_routine_CFmk7g3_rec_cyc3kss,**kwargs)
#
#
#
# def osdanmk7g3_rec_cyc3ksasbdrftfr_ocr_routine(**kwargs):
#     return osdanmk7g3_rec_cyc3ks_family_ocr_routine(neko_HDOS2C_routine_CFmk7g3_rec_cyc3ksasbd_cycrftfr,**kwargs);
# def osdanmk7g3_rec_cyc3ksasbdrfndtfr_ocr_routine(**kwargs):
#     return osdanmk7g3_rec_cyc3ks_family_ocr_routine(neko_HDOS2C_routine_CFmk7g3_rec_cyc3ksasbd_cycrfndtfr,**kwargs)