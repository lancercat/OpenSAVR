from neko_sdk.MJT.neko_abstract_jtr import neko_modular_joint_training_semipara
from configs import dan_single_model_train_cfg
from neko_sdk.environment.root import find_data_root;

import sys

if __name__ == '__main__':
    if(len(sys.argv)>1):
        bs=int(sys.argv[1]);
    else:
        bs=48
    trainer=neko_modular_joint_training_semipara(
        dan_single_model_train_cfg(
            "jtrmodels",
            find_data_root(),
            "../logs/",
            200,
            bsize=bs,
            itrk="Top Nep"
        )
    );
    # with torch.autograd.detect_anomaly():
    trainer.train(None,flag=False);
