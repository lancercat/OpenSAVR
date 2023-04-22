import os
from configs import model_mod_cfg as modcfg
from configs import TAG
from functools import partial

from neko_2022_soai_zero.configs.routines.osdanmk7recon_eval_routine_cfg import osdanmk7r_eval_routine_cfg,arm_base_task_default2ar
from neko_2021_mjt.dss_presets.dual_no_lsct_32 import get_eval_dss_gzsl,get_eval_dss_osr,get_eval_dss_gosr,get_eval_dss_ostr;
def dan_mjst_eval_cfg(evalfn,tag,
        save_root,dsroot,
        log_path,iterkey):

    if(log_path):
        epath=os.path.join(log_path, tag);
    else:
        epath=None;
    te_meta_path_chsjap, chs_eval_ds,has_rej=evalfn(dsroot,25,30,lhw=[32,128]);
    task_dict = {}
    # task_dict = arm_base_task_default2(task_dict, "base_mjst_", osdanmk7_eval_routine_cfg, 25,
    #                                      te_meta_path_mjst, mjst_eval_ds, log_path);
    task_dict = arm_base_task_default2ar(task_dict, TAG+"chs_", osdanmk7r_eval_routine_cfg, 30,
                                         te_meta_path_chsjap, chs_eval_ds,
                                         log_path,measure_rej=has_rej);
    return \
    {
        "root": save_root,
        "iterkey": iterkey, # something makes no sense to start fresh
        "modules": modcfg(None,None,25,30),
        "export_path":epath,
        "tasks":task_dict
    }

dan_open_all={
    "OSR": partial(dan_mjst_eval_cfg, get_eval_dss_osr, "OSR"),
    "GZSL":partial(dan_mjst_eval_cfg,get_eval_dss_gzsl,"GZSL"),
    "GOSR": partial(dan_mjst_eval_cfg, get_eval_dss_gosr,"GOSR"),
    "OSTR": partial(dan_mjst_eval_cfg, get_eval_dss_ostr,"OSTR"),

}
