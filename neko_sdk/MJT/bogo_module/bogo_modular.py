from neko_sdk.MJT.bogo_module.servant_module import neko_stand_basic;


class neko_bogo_modular:
    def __init__(this,forwardable):
        # you should never have a
        this.model=forwardable;
        this.save_each=-9
    # forked stands can only do forward. Fancy controls can yield weird race conditions.

    def train(this,training=True):
        pass;

    def __call__(this, *args, **kwargs):
        return this.model(*args,**kwargs);
    def detach(this):
        this.model.detach();
    def attach(this):
        this.model.attach();
    def dnattach_fwd(this, *args, **kwargs):
        this.detach();
        ret=this(*args,**kwargs);
        this.attach();
        return ret;
    def replicate(this, devices):
        stands=this.model.replicate(devices);
        this.stands = [neko_stand_basic(stand) for stand in stands]
        return this.stands;

    def get_torch_modular_dict(this):
        try:
            return this.model.get_torch_modular_dict();
        except:
            return None;
    # does nothing
    def eval(this):
        pass;
    def normgrad(this):
        pass;

    def zero_grad(this):
        pass;

    def load(this,itrkey):
        pass;
    def cuda(this):
        pass;
    def save(this,nEpoch):
        pass;
    def save_if_needed(this,nEpoch,batch_idx):
        pass;
