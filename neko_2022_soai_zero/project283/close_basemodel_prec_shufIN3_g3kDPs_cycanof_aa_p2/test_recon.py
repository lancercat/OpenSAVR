from eval_configs_recon import dan_mjst_eval_cfg
from configs import TAG
from neko_2022_soai_zero.do_experiments import do_experiments
from neko_sdk.environment.copy import copymodel
from neko_sdk.environment.hosts import get_dev_meta

import sys

if __name__ == '__main__':    
    if(len(sys.argv)<3):
        DEV="DEV5";
        DROOT="/run/media/lasercat/ssddata/all_283/";
        MNAME=__file__.split("/")[-2];
    else:
        DEV=sys.argv[1];
        DROOT = sys.argv[2];
        MNAME = sys.argv[3];
    copymodel(get_dev_meta()[DEV],DROOT, MNAME, "project283")
    do_experiments(dan_mjst_eval_cfg,DROOT,MNAME,"_E4",DEV,tag=TAG);


