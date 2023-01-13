
from neko_sdk.MJT.default_config import get_default_model
from neko_2022_soai_zero.modules.neko_shuf_mod_crop_and_combine import neko_rand_shuf
def get_chunk_shuf(arg_dict,path,optim_path=None):
    args = {"rchunk":arg_dict["rchunk"],
            "cchunk":arg_dict["cchunk"],
    }
    return get_default_model(neko_rand_shuf, args, path, arg_dict["with_optim"], optim_path);

def config_chunk_shuf(rchunk=2,cchunk=2):
    return \
    {
        "modular": get_chunk_shuf,
        "save_each": 20000,
        "args":
            {
                "rchunk":rchunk,
                "cchunk":cchunk,
                # Not like we need it for evaluation, but we still keep it in case of fine-tuning.
                "with_optim": False
            },
    }
