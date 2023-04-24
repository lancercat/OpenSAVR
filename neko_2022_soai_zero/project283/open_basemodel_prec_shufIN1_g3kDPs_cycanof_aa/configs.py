
from neko_2022_soai_zero.configs.routines.arm_water_routine import arm_water2_routine
from neko_2022_soai_zero.configs.loadouts.protorec_v2 import arm_v2IN_prec_shuf_module_setDPS_cyc3
from neko_2022_soai_zero.project283.data_cfg import get_oss_training_protocol
from neko_2022_soai_zero.configs.routines.osdanmk7g3_water2_routine_cfg import osdanmk7g3_rec_cyc3ksanfr_ocr_routine
TAG="shuf3_"
def model_mod_cfg(tr_meta_path_chs,tr_meta_path_mjst,maxT_mjst,maxT_chs):
    capacity=256;
    feat_ch=512;
    mods={};
    mods=arm_v2IN_prec_shuf_module_setDPS_cyc3(mods,TAG+"chs_",maxT_chs,capacity,feat_ch,tr_meta_path_chs,wemb=0,shuf_speed=1,recon_speed=1,cycspeed=1,cycaa=True);
    return mods;

def dan_single_model_train_cfg(save_root,dsroot,
                               log_path,log_each,itrk= "Top Nep",bsize=48,tvitr=200000):
    train_joint_ds,tr_meta_path_chs,maxT_chs,task_dict=get_oss_training_protocol(dsroot,tag=TAG+"chs_",log_path=log_path)
    routines = {};
    routines = arm_water2_routine(routines, TAG+"chs_", osdanmk7g3_rec_cyc3ksanfr_ocr_routine, maxT_chs, log_path,
                                  log_each, "dan_chs_");

    return \
        {
            "root": save_root,
            "val_each": 10000,
            "vitr": 200000,
            "vepoch": 2,
            "iterkey": itrk,  # something makes no sense to start fresh
            "dataloader_cfg":train_joint_ds,
            # make sure the unseen characters are unseen.
            "modules": model_mod_cfg(tr_meta_path_chs,None, 0,maxT_chs),
            "routine_cfgs": routines,
            "tasks": task_dict,
        }