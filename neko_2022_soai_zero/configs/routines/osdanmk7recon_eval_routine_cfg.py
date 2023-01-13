from neko_2022_soai_zero.routines.ocr_routine.mk7g3.osdan_eval_routine_mk7g3_rec import neko_HDOS2C_eval_routine_CFmk7g3_rec,neko_HDOS2C_eval_routine_CFmk7g3_rec_anc
from neko_sdk.MJT.eval_tasks.dan_eval_tasks import neko_odan_eval_tasks
def osdanmk7r_eval_routine_cfg(sampler_name,prototyper_name,feature_extractor_name,
                         CAMname,seq_name,pred_name,loss_name,prec_loss_name,frec_loss_name,shuf_recon_name,label_name,image_name,beacon_name,bmask_name,log_path,name,maxT,measure_rej=False):
    return \
    {
        "name":name,
        "maxT": maxT,
        "routine":neko_HDOS2C_eval_routine_CFmk7g3_rec,
        "mod_cvt_dicts":
        {
            "sampler":sampler_name,
            "prototyper":prototyper_name,
            "feature_extractor":feature_extractor_name,
            "CAM":CAMname,
            "seq": seq_name,
            "preds":pred_name,
            "losses":loss_name,
            "p_recon":prec_loss_name,
            "f_recon":frec_loss_name,
            "shuf_recon":shuf_recon_name,
        },
        "inp_cvt_dicts":
        {
            "label": label_name,
            "image": image_name,
            "proto": "proto",
            # "beacon": beacon_name,
            #            "semb": "semb",
            "plabel": "plabel",
            "tdict": "tdict",
        },
        "measure_rej":measure_rej,
        "log_path":log_path,
    };
def osdanmk7ar_eval_routine_cfg(sampler_name,prototyper_name,feature_extractor_name,
                         CAMname,seq_name,pred_name,loss_name,prec_loss_name,frec_loss_name,label_name,image_name,beacon_name,bmask_name,log_path,name,maxT,measure_rej=False):
    return \
    {
        "name":name,
        "maxT": maxT,
        "routine":neko_HDOS2C_eval_routine_CFmk7g3_rec_anc,
        "mod_cvt_dicts":
        {
            "sampler":sampler_name,
            "prototyper":prototyper_name,
            "feature_extractor":feature_extractor_name,
            "CAM":CAMname,
            "seq": seq_name,
            "preds":pred_name,
            "losses":loss_name,
            "p_recon":prec_loss_name,
            "f_recon":frec_loss_name,
        },
        "inp_cvt_dicts":
        {
            "label": label_name,
            "image": image_name,
            "proto": "proto",
            "beacon": beacon_name,
            #            "semb": "semb",
            "plabel": "plabel",
            "tdict": "tdict",
        },
        "measure_rej":measure_rej,
        "log_path":log_path,
    };



def arm_base_eval_routine2ar(srcdst,tname,prefix,routine_type,log_path,maxT,measure_rej=False,override_dict=None):
    kvargs = {"prototyper_name": prefix + "prototyper",
              "sampler_name": prefix + "Latin_62_sampler",
              "feature_extractor_name": prefix + "feature_extractor_cco",
              "CAMname": prefix + "TA",
              "seq_name": prefix + "DTD",
              "pred_name": [prefix + "pred"],
              "loss_name": [prefix + "loss_cls_emb"],
              "prec_loss_name": prefix + "p_recon",
              "frec_loss_name": prefix + "f_recon",
              "shuf_recon_name": prefix+"shuf_recon",
              "image_name": "image",
              "label_name": "label",
              "beacon_name":"beacon",
              "bmask_name":"bmask",
              "log_path": log_path,
              "name": prefix + tname,
              "maxT": maxT,
              "measure_rej": measure_rej, }
    if(override_dict is not None):
        for k in override_dict:
            kvargs[k]=override_dict[k];
    return routine_type(
        **kvargs
    );


def arm_base_task_default2ar(srcdst,prefix,routine_type,maxT,te_meta_path,datasets,log_path,name="close_set_benchmarks",measure_rej=False,override_dict=None):
    te_routine={};
    te_routine=arm_base_eval_routine2ar(te_routine,"close_set_benchmark",
                                      prefix,routine_type,log_path,maxT,measure_rej=measure_rej,override_dict=override_dict)
    if(override_dict is not None):
        if("prototyper_name" in override_dict):
            pengine=override_dict["prototyper_name"];
        else:
            pengine=prefix + "prototyper"
    else:
        pengine = prefix + "prototyper"

    srcdst[prefix+name]={
        "type": neko_odan_eval_tasks,
        "protoname":pengine,
        "temeta":
            {
                "meta_path": te_meta_path,
                "case_sensitive": False,
            },
        "datasets":datasets,
        "routine_cfgs": te_routine,
    }
    return srcdst