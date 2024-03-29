import os
from configs import model_mod_cfg as modcfg
from configs import TAG


from neko_2022_soai_zero.configs.routines.osdanmk7recon_eval_routine_cfg import osdanmk7r_eval_routine_cfg,arm_base_task_default2ar
from neko_2021_mjt.configs.data.chs_jpn_data import get_chs_HScqa,get_eval_jpn_color,get_chs_tr_meta,get_jpn_te_meta
from neko_2021_mjt.dss_presets.dual_no_lsct_32 import get_eval_dss;
def dan_mjst_eval_cfg(
        save_root,dsroot,
        log_path,iterkey,maxT=30):

    if(log_path):
        epath=os.path.join(log_path, "closeset_benchmarks");
    else:
        epath=None;
    te_meta_path_chsjap, te_meta_path_mjst, mjst_eval_ds, chs_eval_ds=get_eval_dss(dsroot,25,30,lhw=[32,128]);
    task_dict = {}
    # task_dict = arm_base_task_default2(task_dict, "base_mjst_", osdanmk7_eval_routine_cfg, 25,
    #                                      te_meta_path_mjst, mjst_eval_ds, log_path);
    task_dict = arm_base_task_default2ar(task_dict, TAG+"mjst_", osdanmk7r_eval_routine_cfg, 25,
                                         te_meta_path_mjst, mjst_eval_ds,
                                         log_path);


    return \
    {
        "root": save_root,
        "iterkey": iterkey, # something makes no sense to start fresh
        "modules": modcfg(None,None,25,30),
        "export_path":epath,
        "tasks":task_dict
    }
