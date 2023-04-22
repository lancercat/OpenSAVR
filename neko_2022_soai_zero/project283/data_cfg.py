from neko_2021_mjt.configs.routines.ocr_routines.mk7.osdanmk7_routine_cfg import osdanmk7_eval_routine_cfg
from neko_2021_mjt.dss_presets.dual_no_lsct_32 import get_dssscht,get_eval_dss_jk,get_eval_dss_close,get_dss_close;
from neko_2021_mjt.configs.loadouts.base_module_set import arm_base_task_default2


def get_oss_training_protocol(dsroot,tag,log_path,bsize=48):
    maxT_chs=30;
    tr_meta_path_chs, train_joint_ds = get_dssscht(dsroot, maxT_chs, bsize);
    te_meta_path_jpn,te_meta_path_kr,jap_eval_ds,kr_eval_ds=get_eval_dss_jk(dsroot,maxT_chs);

    task_dict = {}
    # task_dict = arm_base_task_default2(task_dict, "base_mjst_", osdanmk7_eval_routine_cfg, maxT_mjst, te_meta_path_mjst,mjst_eval_ds , log_path);
    task_dict = arm_base_task_default2(task_dict, tag, osdanmk7_eval_routine_cfg, maxT_chs, te_meta_path_jpn,
                                       jap_eval_ds,
                                       log_path,name="GZSL-CHS-JP");
    # task_dict = arm_base_task_default2(task_dict, tag, osdanmk7_eval_routine_cfg, maxT_chs, te_meta_path_kr,
    #                                    kr_eval_ds,
    #                                    log_path,name="GZSL-CHS-KR");
    return train_joint_ds,tr_meta_path_chs,maxT_chs,task_dict;
def get_close_training_protocol(dsroot,tag,log_path,bsize=48):
    maxT_mjst=25;
    tr_meta_path_mjst, te_meta_path_mjst, mjst_eval_ds, train_joint_ds=get_dss_close(dsroot,maxT_mjst,bsize);
    task_dict = {}
    # task_dict = arm_base_task_default2(task_dict, "base_mjst_", osdanmk7_eval_routine_cfg, maxT_mjst, te_meta_path_mjst,mjst_eval_ds , log_path);
    task_dict = arm_base_task_default2(task_dict, tag, osdanmk7_eval_routine_cfg, maxT_mjst, te_meta_path_mjst,
                                       mjst_eval_ds,
                                       log_path,name="MJST");
    # task_dict = arm_base_task_default2(task_dict, tag, osdanmk7_eval_routine_cfg, maxT_chs, te_meta_path_kr,
    #                                    kr_eval_ds,
    #                                    log_path,name="GZSL-CHS-KR");
    return train_joint_ds,te_meta_path_mjst,maxT_mjst,task_dict;

