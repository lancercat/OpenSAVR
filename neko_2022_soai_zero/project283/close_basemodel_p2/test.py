from eval_configs import dan_mjst_eval_cfg

if __name__ == '__main__':
    import sys
    if(len(sys.argv)<2):
        argv = ["Meeeeooooowwww",
                "/run/media/lasercat/ssddata/mc-285/open_mc_basemodel_protorec/jtrmodels/",
                "_E0",
                "/run/media/lasercat/ssddata/mc-285/open_mc_basemodel_protorec/jtrmodels/",
                ]
    else:
        argv=sys.argv;
    from neko_sdk.MJT.lanuch_std_test import launchtest
    launchtest(argv,dan_mjst_eval_cfg)
