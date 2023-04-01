from eval_configs_m import dan_mjst_eval_cfg

if __name__ == '__main__':
    import sys
    if(len(sys.argv)<2):
        argv = ["Meeeeooooowwww",
                "/home/lasercat/ssddata/cvpr22_candidata/7hdt/",
                "_E1",
                "/home/lasercat/ssddata/cvpr22_candidata/7hdt/",
                ]
    else:
        argv=sys.argv;
    from neko_sdk.MJT.lanuch_std_test import launchtest
    launchtest(argv,dan_mjst_eval_cfg)
