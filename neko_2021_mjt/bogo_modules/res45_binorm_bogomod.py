class neko_res45_binorm_bogo_stand:
    def __init__(this, args, mod_dict):
        this.container = mod_dict[args["container"]];
        this.name = args["name"];
        this.bnname = args["name"].replace("res", "bn");
        this.model = mod_dict[args["container"]].model.refresh_bogo()[args["name"]]
    def detach(this):
        this.container.detach();
    def attach(this):
        this.container.attach();
    def __call__(this, *args,**kwargs):
        return this.model(*args,**kwargs);

class neko_res45_binorm_bogo(neko_res45_binorm_bogo_stand):
    def cuda(this):
        pass;
    def replicate(this, devices):
        moddicts = [{} for d in devices];
        args = [{} for a in devices];
        this.stands = []
        for did in range(len(devices)):
            args[did]["name"] = this.name;
            args[did]["bnname"] = this.bnname;

            moddicts[did]["container"] = this.container.stands[did];
            args[did]["container"] = "container";
            this.stands.append(neko_res45_binorm_bogo_stand(args[did], moddicts[did]))
        return this.stands;

    def freeze(this):
        this.container.eval();
    def freezebn(this):
        this.container.model.freezebnprefix(this.bnname);
    def unfreezebn(this):
        this.container.model.unfreezebnprefix(this.bnname);

    def unfreeze(this):
        this.container.model.train();
    def get_torch_modular_dict(this):
        return this.container.model;

