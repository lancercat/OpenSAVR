# bogomodules are certain combination to the modules, they do not hold parameters
# instead they use whatever armed to the module set.
# Different to routines, they are statically associated to a certain set of modules for speed up.
import torch;
from torch.nn import functional as trnf
# some time you cannot have a container. That's life.
# say u have module a b c d.
# A uses [ac] B uses [ab] C uses [ad]...
# There is no better way than simply put a,b,c,d in a big basket.
class gen3_object_to_feat_abstract:
    def proto_engine(this,clips):
        features = this.backbone(clips)
        if(this.detached_ga):
            A=this.cam([f.detach() for f in features]);
        else:
            A = this.cam(features);
        # A=torch.ones_like(A);
        out_emb=(A*features[-1]).sum(-1).sum(-1)/A.sum(-1).sum(-1);
        return out_emb;
    def proto_engine_debug(this,clips):
        features = this.backbone(clips);
        if(this.detached_ga):
            A=this.cam([f.detach() for f in features]);
        else:
            A = this.cam(features);

        # A=torch.ones_like(A);
        out_emb=(A*features[-1]).sum(-1).sum(-1)/A.sum(-1).sum(-1);
        return out_emb,A;
    def detach(this):
        this.backbone.detach();
        this.cam.detach();
    def attach(this):
        this.backbone.model.attach();
        this.cam.attach();

    def __init__(this,args,moddict):
        this.detached_ga=args["detached_ga"];
        this.backbone=moddict[args["backbone"]];
        this.cam=moddict[args["cam"]];

# stand can only call the modules.
# only (bogo)modules can change their statues like training, evaluation, accpetance of gradient.
class prototyper_gen3_stand(gen3_object_to_feat_abstract):
    def __init__(this, args, moddict):
        super(prototyper_gen3_stand, this).__init__(args, moddict);
        this.detached_ga=args["detached_ga"];
        this.force_proto_shape = args["force_proto_shape"];
        this.capacity = args["capacity"];
        # sometimes we do not need sp protos.
        if (args["sp_proto"] in moddict):
            this.sp = moddict[args["sp_proto"]]
        else:
            this.sp = None;
        if ("device" not in args):
            this.device_indicator = "cuda";
        else:
            this.device_indicator = args["device"];
        if (args["drop"]):
            if (args["drop"] == "NEP_reuse_NEP"):
                this.drop = this.backbone.model.model.dropper;
            else:
                this.drop = moddict[args["drop"]];
        else:
            this.drop=None;
    def forward_stub(this,normprotos, rot=0,engine=None):
        # pimage=torch.cat(normprotos).to(this.dev_ind.device);
        pimage = torch.cat(normprotos).contiguous().to(this.device_indicator);
        # print(this.device_indicator);
        if (this.force_proto_shape is not None and pimage.shape[-1] != this.force_proto_shape):
            pimage = trnf.interpolate(pimage, [this.force_proto_shape, this.force_proto_shape], mode="bilinear");
        if (rot > 0):
            pimage = torch.rot90(pimage, rot, [2, 3]);

        if (pimage.shape[1] == 1):
            pimage = pimage.repeat([1, 3, 1, 1]);
        return engine(pimage);


    def forward(this, normprotos, rot=0, use_sp=True):
        if (len(normprotos) <= this.capacity):
            if (use_sp):
                spproto, _ = this.sp();
                proto = [spproto, this.forward_stub(normprotos,rot,engine=this.proto_engine)];
            else:
                proto = [this.forward_stub(normprotos,rot,engine=this.proto_engine)]
        else:
            if (use_sp):
                spproto, _ = this.sp();
                proto = [spproto];
            else:
                proto = [];
            chunk = this.capacity // 4;
            for s in range(0, len(normprotos), chunk):
                batchp=this.forward_stub(normprotos[s:s + chunk],rot,engine=this.proto_engine);
                proto.append(batchp)
        allproto=torch.cat(proto).contiguous();
        if (this.drop):
            allproto = this.drop(allproto);
        allproto = trnf.normalize(allproto, dim=1, eps=0.0009);

        return allproto;
    def forward_debug(this, normprotos, rot=0, use_sp=True):
        if (len(normprotos) <= this.capacity):
            if (use_sp):
                spproto, _ = this.sp();
                p,a=this.forward_stub(normprotos, rot,engine=this.proto_engine_debug);
                proto = [spproto,p];
                atts=[a.detach().cpu()];
            else:
                p,a=this.forward_stub(normprotos, rot,engine=this.proto_engine_debug);
                proto = [p]
                atts = [a.detach().cpu()]
        else:
            if (use_sp):
                spproto, _ = this.sp();
                proto = [spproto];
            else:
                proto = [];
            atts = [];
            chunk = this.capacity // 4;
            for s in range(0, len(normprotos), chunk):
                batchp,batcha = this.forward_stub(normprotos[s:s + chunk], rot,engine=this.proto_engine_debug);

                proto.append(batchp);
                atts.append(batcha.detach().cpu())
        allproto=torch.cat(proto);
        if (this.drop):
            allproto = this.drop(allproto);
        allproto = trnf.normalize(allproto, dim=-1, eps=0.0009);
        allatt=torch.cat(atts);

        return allproto.contiguous(),allatt;
    def __call__(this, *args, **kwargs):
        return this.forward(*args, **kwargs);




class prototyper_gen3:
    def get_stand(this,args,moddict):
        return prototyper_gen3_stand(args,moddict);
    # The info are kept to later replicate stands.
    # The api is optimized to support multi-gpu training
    # however the users are responsible to avoid race condition when trying to freeze and unfreeze bn and etc.
    # we ourself do not use multi-gpu setup since we do not currently benefit from multi-gpu training, but since someone requested...
    def __init__(this, args, moddict):
        this.detached_ga=args["detached_ga"];
        this.force_proto_shape = args["force_proto_shape"];
        this.capacity = args["capacity"];
        # sometimes we do not need sp protos.
        if (args["sp_proto"] in moddict):
            this.sp = moddict[args["sp_proto"]]
        else:
            this.sp = None;
        this.backbone = moddict[args["backbone"]];
        this.cam = moddict[args["cam"]];
        if ("device" not in args):
            this.device_indicator = "cuda";
        else:
            this.device_indicator = args["device"];
        if (args["drop"]):
            if(args["drop"]=="NEP_reuse_NEP"):
                this.drop=this.backbone.model.model.dropper;
            else:
                this.drop = moddict[args["drop"]];
        else:
            this.drop = None;
        this.stand=this.get_stand(args,moddict);

    def __call__(this, *args, **kwargs):
        return this.stand(*args,**kwargs);
    def forward_debug(this, *args, **kwargs):
        return this.stand.forward_debug(*args,**kwargs);

    def replicate(this,devices):
        moddicts=[{} for d in devices];
        args=[{} for a in devices];
        this.stands=[]
        for did in range(len(devices)):
            args[did]["capacity"]=this.capacity;
            args[did]["force_proto_shape"]=this.force_proto_shape;
            args[did]["device"]=devices[did];

            moddicts[did]["backbone"]=this.backbone.stands[did];
            args[did]["backbone"]="backbone";

            moddicts[did]["cam"]=this.cam.stands[did];
            args[did]["cam"]="cam";

            if(this.sp):
                moddicts[did]["sp_proto"] = this.cam.stands[did];
                args[did]["sp_proto"] = "sp_proto";
            else:
                args[did]["sp_proto"]="TopNEP"
            if(this.drop):
                moddicts[did]["drop"] = this.cam.stands[did];
                args[did]["drop"] = "drop";
            else:
                args[did]["drop"]=None;

            this.stands.append(this.get_stand(args[did],moddicts[did]))
        return this.stands;

    def freeze(this):
        this.backbone.model.freeze();
        if(this.sp is not None):
            this.sp.eval();
        this.cam.eval();

    def freeze_bb_bn(this):
        this.backbone.model.freezebn();

    def unfreeze_bb_bn(this):
        this.backbone.model.unfreezebn();


    def unfreeze(this):
        this.backbone.model.unfreeze();
        if(this.sp is not None):
            this.sp.train();
        this.cam.train();

    def cuda(this):
        this.device_indicator="cuda";
