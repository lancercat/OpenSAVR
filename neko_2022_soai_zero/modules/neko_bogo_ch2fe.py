from neko_sdk.MJT.bogo_module.prototype_gen3 import gen3_object_to_feat_abstract
# Yup, it's that simple----for now.
class neko_recon_fe(gen3_object_to_feat_abstract):
    def cuda(*arge):
        pass;

    def __call__(this, proto):
        return this.proto_engine(proto.repeat(1,3,1,1));
