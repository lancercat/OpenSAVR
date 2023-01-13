from neko_sdk.MJT.bogo_module.prototype_gen3 import prototyper_gen3

def config_prototyper_gen3(sp_proto, backbone, cam, drop=None, capacity=512, force_proto_shape=None,detached_ga=False):
    return {
        "bogo_mod": prototyper_gen3,
        "args":
        {
            "capacity":capacity,
            "sp_proto":sp_proto,
            "backbone":backbone,
            "cam":cam,
            "drop":drop,
            "force_proto_shape":force_proto_shape,
            "detached_ga":detached_ga,
        }
    }
