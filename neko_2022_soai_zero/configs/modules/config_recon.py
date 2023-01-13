from neko_sdk.MJT.default_config import get_default_model
from neko_sdk.TLS.reconstruction.neko_prototype_reconstruction import neko_dcgan_reconstructionN_insnorm,neko_masked_l2_loss

def get_dcganN_insnorm(arg_dict,path,optim_path=None):
    args={
        "psize": arg_dict["ifc"],"gsize":arg_dict["gsize"]
    }
    return get_default_model(neko_dcgan_reconstructionN_insnorm, args, path, arg_dict["with_optim"], optim_path);
def config_dcganN_insnorm(ifc,gsize=64):
    return \
    {
        "modular": get_dcganN_insnorm,
        "save_each": 20000,
        "args":
            {
                "ifc":ifc,
                "gsize":gsize,
                # Not like we need it for evaluation, but we still keep it in case of fine-tuning.
                "with_optim": True
            },
    }
def get_masked_l2(arg_dict,path,optim_path=None):
    args = {"speed":arg_dict["speed"],
    }
    return get_default_model(neko_masked_l2_loss, args, path, arg_dict["with_optim"], optim_path);

def config_masked_l2(speed=1):
    return \
    {
        "modular": get_masked_l2,
        "save_each": 20000,
        "args":
            {
                "speed":speed,
                # Not like we need it for evaluation, but we still keep it in case of fine-tuning.
                "with_optim": False
            },
    }
#
# from neko_sdk.MJT.default_config import get_default_model
# from neko_sdk.TLS.reconstruction.neko_prototype_reconstruction import \
#     neko_btl2_reconstruction_loss,neko_masked_l2_loss,neko_masked_l2_loss_reduction,\
#     neko_dcgan_reconstruction,neko_dcgan_reconstructionN,neko_dcgan_reconstructionN_insnorm
# from neko_sdk.TLS.reconstruction.neko_masked_l2_loss_reduction_g2 import neko_masked_l2_loss_reductiong2
#
# def get_dcgan(arg_dict,path,optim_path=None):
#     args={
#         "psize": arg_dict["ifc"],"gsize":arg_dict["gsize"]
#     }
#     return get_default_model(neko_dcgan_reconstruction, args, path, arg_dict["with_optim"], optim_path);
# def config_dcgan(ifc,gsize=64):
#     return \
#     {
#         "modular": get_dcgan,
#         "save_each": 20000,
#         "args":
#             {
#                 "ifc":ifc,
#                 "gsize":gsize,
#                 # Not like we need it for evaluation, but we still keep it in case of fine-tuning.
#                 "with_optim": True
#             },
#     }
#
# def get_dcganN(arg_dict,path,optim_path=None):
#     args={
#         "psize": arg_dict["ifc"],"gsize":arg_dict["gsize"]
#     }
#     return get_default_model(neko_dcgan_reconstructionN, args, path, arg_dict["with_optim"], optim_path);
# def config_dcganN(ifc,gsize=64):
#     return \
#     {
#         "modular": get_dcganN,
#         "save_each": 20000,
#         "args":
#             {
#                 "ifc":ifc,
#                 "gsize":gsize,
#                 # Not like we need it for evaluation, but we still keep it in case of fine-tuning.
#                 "with_optim": True
#             },
#     }
#
# def get_dcganN_insnorm(arg_dict,path,optim_path=None):
#     args={
#         "psize": arg_dict["ifc"],"gsize":arg_dict["gsize"]
#     }
#     return get_default_model(neko_dcgan_reconstructionN_insnorm, args, path, arg_dict["with_optim"], optim_path);
# def config_dcganN_insnorm(ifc,gsize=64):
#     return \
#     {
#         "modular": get_dcganN_insnorm,
#         "save_each": 20000,
#         "args":
#             {
#                 "ifc":ifc,
#                 "gsize":gsize,
#                 # Not like we need it for evaluation, but we still keep it in case of fine-tuning.
#                 "with_optim": True
#             },
#     }

# def get_masked_l2_reduction(arg_dict,path,optim_path=None):
#     args = {"speed":arg_dict["speed"],
#     }
#     return get_default_model(neko_masked_l2_loss_reduction, args, path, arg_dict["with_optim"], optim_path);
#
# def config_masked_l2_reduction(speed=1):
#     return \
#     {
#         "modular": get_masked_l2_reduction,
#         "save_each": 20000,
#         "args":
#             {
#                 "speed":speed,
#                 # Not like we need it for evaluation, but we still keep it in case of fine-tuning.
#                 "with_optim": False
#             },
#     }
#
#
# def get_masked_l2_reduction_gen2(arg_dict,path,optim_path=None):
#     args = {"speed":arg_dict["speed"],
#     }
#     return get_default_model(neko_masked_l2_loss_reductiong2, args, path, arg_dict["with_optim"], optim_path);
#
# def config_masked_l2_reduction_gen2(speed=1):
#     return \
#     {
#         "modular": get_masked_l2_reduction_gen2,
#         "save_each": 20000,
#         "args":
#             {
#                 "speed":speed,
#                 # Not like we need it for evaluation, but we still keep it in case of fine-tuning.
#                 "with_optim": False
#             },
#     }
#
#
#
# def get_dgrl(arg_dict,path,optim_path=None):
#     args={
#         "psize":arg_dict["ifc"],
#     }
#     return get_default_model(neko_btl2_reconstruction_loss, args, path, arg_dict["with_optim"], optim_path);
#
# def config_dgrl(ifc):
#     return \
#     {
#         "modular": get_dgrl,
#         "save_each": 20000,
#         "args":
#             {
#                 "ifc":ifc,
#                 # Not like we need it for evaluation, but we still keep it in case of fine-tuning.
#                 "with_optim": True
#             },
#     }
#
# def get_dgrlX(arg_dict,path,optim_path=None):
#     args={
#         "psize":arg_dict["ifc"],
#     }
#     return get_default_model(neko_btl2_reconstruction_loss, args, path, arg_dict["with_optim"], optim_path);
#
# def config_dgrlX(ifc):
#     return \
#     {
#         "modular": get_dgrlX,
#         "save_each": 20000,
#         "args":
#             {
#                 "ifc":ifc,
#                 # Not like we need it for evaluation, but we still keep it in case of fine-tuning.
#                 "with_optim": True
#             },
#     }
