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
