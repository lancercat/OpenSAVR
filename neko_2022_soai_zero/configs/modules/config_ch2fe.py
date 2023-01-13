from neko_2022_soai_zero.modules.neko_bogo_ch2fe import neko_recon_fe

def config_ch2fe(backbone, cam, force_proto_shape=None,detached_ga=False):
    return {
        "bogo_mod": neko_recon_fe,
        "args":
        {
            "backbone":backbone,
            "cam":cam,
            "force_proto_shape":force_proto_shape,
            "detached_ga":detached_ga,
        }
    }