# sampler_name, prototyper_name, feature_extractor_name, seq_name,
#                            CAMname, pred_name,ctx_name, loss_name,dom_loss_name,prec_loss_name,frec_loss_name,
#                            label_name, image_name, log_path, log_each, name, maxT
# srcdst["ctxmod"] = "NEP_skipped_NEP";
# srcdst["dom_mix"] = "NEP_skipped_NEP";
# srcdst["p_recon"] = "NEP_skipped_NEP";
# srcdst["f_recon"] = "NEP_skipped_NEP";
# return srcdst;
def arm_water_routine(srcdst, prefix, routine_type,maxT, log_path, log_each,dsprefix,override_dict=None,rot_id="mjst"):

    kvargs={
        "prototyper_name":prefix+"prototyper",
        "sampler_name":prefix+"Latin_62_sampler",
        "feature_extractor_name":prefix+"feature_extractor_cco",
        "CAMname":prefix+"TA",
        "seq_name":prefix+"DTD",
        "pred_name":[prefix+"pred"],
        "ctx_name": prefix + "ctxmod",
        "loss_name":[prefix+"loss_cls_emb"],
        "dom_loss_name": prefix+"dom_mix",
        "prec_loss_name": prefix+"p_recon",
        "frec_loss_name":  prefix+"f_recon",
        "image_name":dsprefix+"image",
        "label_name":dsprefix+"label",
        "log_path":log_path,
        "log_each":log_each,
        "name":prefix+rot_id,
        "maxT":maxT,}
    if(override_dict is not None):
        for k in override_dict:
            kvargs[k]=override_dict[k];
    srcdst[prefix+rot_id]= routine_type(
        **kvargs
    );

    srcdst[prefix+rot_id]["stream"]=prefix;
    return srcdst;


def arm_water2_routine(srcdst, prefix, routine_type,maxT, log_path, log_each,dsprefix,override_dict=None,rot_id="mjst"):

    kvargs={
        "prototyper_name":prefix+"prototyper",
        "sampler_name":prefix+"Latin_62_sampler",
        "feature_extractor_name":prefix+"feature_extractor_cco",
        "CAMname":prefix+"TA",
        "seq_name":prefix+"DTD",
        "pred_name":[prefix+"pred"],
        "ctx_name": prefix + "ctxmod",
        "loss_name":[prefix+"loss_cls_emb"],
        "dom_loss_name": prefix+"dom_mix",
        "prec_name": prefix + "p_recon",
        "prec_loss_name": prefix+"p_recon_loss",
        "shuf_part_recon_name": prefix + "shuf_part_recon",
        "shuf_part_recon_loss_name": prefix + "shuf_part_recon_loss",
        "proto_part_recon_name": prefix + "proto_part_recon",
        "proto_part_recon_loss_name": prefix + "proto_part_recon_loss",
        "shuf_img_name": prefix+"shuf_img",
        "shuf_recon_name": prefix+"shuf_recon",
        "shuf_recon_loss_name":prefix+"shuf_recon_loss",
        "shuf_sim_loss_name": prefix+"shuf_sim_loss",
        "shuf_proto_name":prefix+"shuf_proto",
        "frec_name":  prefix+"f_recon",
        "inv_loss_name":prefix+"inv_loss",
        "recon_char_fe_name": prefix+"recon_char_fe",
        "recon_char_pred_name": prefix + "recon_char_pred",
        "frec_loss_name": prefix + "f_recon_loss",
        "fpm_recon_loss_name": prefix+"fpm_recon_loss",
        "image_name":dsprefix+"image",
        "label_name":dsprefix+"label",
        "log_path":log_path,
        "log_each":log_each,
        "name":prefix+rot_id,
        "maxT":maxT,}
    if(override_dict is not None):
        for k in override_dict:
            kvargs[k]=override_dict[k];
    srcdst[prefix+rot_id]= routine_type(
        **kvargs
    );

    srcdst[prefix+rot_id]["stream"]=prefix;
    return srcdst;


