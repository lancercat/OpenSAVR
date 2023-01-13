import os
# from neko_2022_soai_zero.visualization.tsne0 import do_tsne
# from dan_based_logparser.compare_results_by_id import compare_files
from neko_2021_mjt.lanuch_std_test import launchtest


def do_experiments(evalcfg,root,method,epoch="_E0",dev="ABLMAT",tag="base_",export_path="/home/lasercat/ssddata/export/"):
    argv = ["Meeeeooooowwww",
            os.path.join(root,dev,method,"jtrmodels"),
            epoch,
            os.path.join(root,dev,method,"jtrmodels"),
            ]
    rawpath=os.path.join(root,dev,method,"jtrmodels/closeset_benchmarks/",tag+"chs_prototyper/JAP_lang/");
    os.makedirs(rawpath,exist_ok=True);
    launchtest(argv, evalcfg,export_path=export_path);
    # compare_files(os.path.join(root,dev),[method],os.path.join(root,dev,method,"dashboard"),4009,[tag]);
    # do_tsne(rawpath);

def do_experiments2(evalcfgs,root,method,epoch="_E0",dev="ABLMAT",tag="base_"):
    from neko_2021_mjt.lanuch_std_test import launchtest
    argv = ["Meeeeooooowwww",
            os.path.join(root,dev,method,"jtrmodels"),
            epoch,
            os.path.join(root,dev,method,"jtrmodels"),
            ]
    for k in evalcfgs:
        #                                                 protocol     MISD model          Dataset
        rawpath=os.path.join(root,dev,method,"jtrmodels/",k,      tag+"chs_prototyper", "JAP_lang/");
        os.makedirs(rawpath,exist_ok=True);
        launchtest(argv, evalcfgs[k]);
        try:
            do_tsne(rawpath);
        except:
            print("tsne error");
        try:
            compare_files(os.path.join(root,dev),[method],os.path.join(root,dev,method,"dashboard"),4009,[tag]);
        except:
            print("no dashboard available")