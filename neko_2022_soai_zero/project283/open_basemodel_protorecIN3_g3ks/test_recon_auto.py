from configs import TAG
from neko_2022_soai_zero.do_experiments import do_experiments2
from neko_sdk.environment.copy import copymodel
from neko_sdk.environment.hosts_home import get_dev_meta
from eval_configs_recon_auto import dan_open_all
import sys

if __name__ == '__main__':
    if(len(sys.argv)<3):
        DEV="ABLMAT";
        DROOT="/run/media/lasercat/ssddata/all_283/";
        MNAME=__file__.split("/")[-2];
    else:
        DEV=sys.argv[1];
        DROOT = sys.argv[2];
        MNAME = sys.argv[3];
    copymodel(get_dev_meta()[DEV],DROOT, MNAME, "project283")
    do_experiments2(dan_open_all,DROOT,MNAME,"_E0",DEV,tag=TAG);


