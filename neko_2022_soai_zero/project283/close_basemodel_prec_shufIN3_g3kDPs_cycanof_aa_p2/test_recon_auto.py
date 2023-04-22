from configs import TAG
from neko_2022_soai_zero.do_experiments import do_experiments2
from neko_sdk.environment.copy import copymodel
from neko_sdk.environment.hosts import get_dev_meta
from eval_configs_recon_auto import dan_close_all
if __name__ == '__main__':
    DEV="DEV3";
    DROOT="/run/media/lasercat/ssddata/all_283/";
    MNAME=__file__.split("/")[-2];
    copymodel(get_dev_meta()[DEV],DROOT, MNAME, "project283")
    do_experiments2(dan_close_all,DROOT,MNAME,"latest",DEV,tag=TAG);


