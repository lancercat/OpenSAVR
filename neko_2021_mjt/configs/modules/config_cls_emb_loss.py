
from neko_2021_mjt.modulars.dan.danloss import osdanloss_clsemb,clsloss
def get_loss_cls_emb2(arg_dict,path,optim_path=None):
    mod=osdanloss_clsemb(arg_dict);
    return mod,None,None;
def config_cls_emb_loss2(wemb=0.3,wrej=0.1,reduction=True,wcls=1):
    return \
    {
        "save_each": 0,
        "modular": get_loss_cls_emb2,
        "args":
            {
                "wcls": wcls,
                "reduction": reduction,
                "wemb": wemb,
                "wrej":wrej
            },
    }

def get_loss_cls(arg_dict,path,optim_path=None):
    mod=clsloss(arg_dict);
    return mod,None,None;
def config_cls_loss(wcls=1):
    return \
    {
        "save_each": 0,
        "modular": get_loss_cls,
        "args":
            {
                "wcls": wcls,
            },
    }
