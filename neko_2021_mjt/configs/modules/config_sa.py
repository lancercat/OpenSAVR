from neko_2021_mjt.modulars.spatial_attention import spatial_attention,spatial_attention_mk3
from neko_sdk.MJT.default_config import get_default_model
def get_sa(arg_dict,path,optim_path=None):
    args={"ifc":arg_dict["num_channels"]};
    # args["num_channels"]=arg_dict["num_channels"]
    return get_default_model(spatial_attention,args,path,arg_dict["with_optim"],optim_path);

def config_sa(feat_ch=512):
    return \
    {
        "modular": get_sa,
        "save_each": 20000,
        "args":
            {
                "with_optim": True,
                "num_channels":feat_ch,
            },
    }
def get_sa_mk3(arg_dict,path,optim_path=None):
    args={"ifc":arg_dict["num_channels"],"n_parts":arg_dict["n_parts"]};
    # args["num_channels"]=arg_dict["num_channels"]
    return get_default_model(spatial_attention_mk3,args,path,arg_dict["with_optim"],optim_path);

def config_sa_mk3(feat_ch=512,n_parts=1):
    return \
    {
        "modular": get_sa_mk3,
        "save_each": 20000,
        "args":
            {
                "with_optim": True,
                "n_parts":n_parts,
                "num_channels":feat_ch,
            },
    }
