from neko_2022_soai_zero.routines.ocr_routine.mk7g3.osdan_routine_mk7g3_rec import neko_HDOS2C_routine_CFmk7g3_rec
def osdanmk7g3_rec_ocr_routine(sampler_name, prototyper_name, feature_extractor_name, seq_name,
                           CAMname, pred_name,ctx_name, loss_name,dom_loss_name,prec_loss_name,frec_loss_name,
                           label_name, image_name, log_path, log_each, name, maxT):

    dic={
        "maxT": maxT,
        "name":name,
        "routine":neko_HDOS2C_routine_CFmk7g3_rec,
        "mod_cvt_dicts":
        {
            "sampler": sampler_name,
            "prototyper":prototyper_name,
            "feature_extractor":feature_extractor_name,
            "CAM":CAMname,
            "seq": seq_name,
            "preds":pred_name,
            "losses":loss_name,
            "ctxmod": ctx_name,
            "dom_mix":dom_loss_name,
            "p_recon":prec_loss_name,
            "f_recon":frec_loss_name,
        },
        "inp_cvt_dicts":
        {
            "label":label_name,
            "image":image_name,
        },
        "log_path":log_path,
        "log_each":log_each,
    }

    return dic;
