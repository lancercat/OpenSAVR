from neko_2021_mjt.configs.modules.config_sp import config_sp_prototyper
from neko_2021_mjt.configs.bogo_modules.config_back_sharing_prototyper import \
    config_prototyper_gen3

def arm_shared_prototyper_np(srcdst,prefix,capacity, feat_ch,fe_name,cam_name,use_sp=True,force_proto_shape=None,detached_ga=False,nameoverride=None,drop=False):
    if(use_sp):
        srcdst[prefix+"sp_proto"]=config_sp_prototyper(feat_ch,use_sp=use_sp);
    if(nameoverride is None):
        nameoverride=prefix+"prototyper";
    srcdst[nameoverride]=config_prototyper_gen3(
        prefix+"sp_proto",
        fe_name,
        cam_name,
        drop,
        capacity,
        force_proto_shape,
        detached_ga=detached_ga,
    )
    return srcdst;
