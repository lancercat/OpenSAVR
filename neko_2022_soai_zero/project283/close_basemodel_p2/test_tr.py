from neko_sdk.MJT.neko_abstract_jtr import neko_abstract_modular_joint_eval
from eval_configs import dan_mjst_eval_cfg_d
from neko_sdk.environment.root import find_data_root;

if __name__ == '__main__':
    import sys

    if (len(sys.argv) > 1):
        export_path = sys.argv[1];
        if (export_path == "None"):
            export_path = None;
    else:
        export_path = "/home/lasercat/ssddata/export/";

    trainer=neko_abstract_modular_joint_eval(
        dan_mjst_eval_cfg_d(
            "jtrmodels",
            find_data_root(),
            export_path,
            "_E0_I120000",
        ),100
    );
    # hack on rejection bias.
    # trainer.modular_dict["DTD"].model.UNK_SCR = torch.nn.Parameter(
    #     torch.zeros_like(trainer.modular_dict["DTD"].model.UNK_SCR))

    # with torch.autograd.detect_anomaly():
    trainer.valt(9,9);
