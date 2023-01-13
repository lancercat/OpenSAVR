import random
class neko_abstract_sampler:
    def setup_sampler(this,sampler_args):
        if sampler_args is None:
            max_match_size = 512;
            val_frac = 0.8;
            neg_servant = True;
        else:
            max_match_size = sampler_args["max_batch_size"];
            val_frac = sampler_args["val_frac"];
            neg_servant = sampler_args["neg_servant"];
        this.max_batch_size = max_match_size;
        this.val_frac = val_frac;
        this.neg_servant = neg_servant;
    def grab_cluster(this,meta,ch):
        chid=meta.label_dict[ch];
        ret={chid};
        if meta.masters_share:
            ret.add(meta.masters[chid]);
            ret=ret.union(meta.servants[meta.masters[chid]]);
        return ret;


    def get_sampled_ids(this,meta,plain_chars_in_data):
        cntval = int(len(plain_chars_in_data) * this.val_frac);
        cntval = min(this.max_batch_size - meta.sp_cnt, cntval);
        trchs=set();
        related_chars_in_data=set();
        random.shuffle(plain_chars_in_data);
        # make sure no missing centers--
        # or it may enforce "A" to look like "a" encoded by proto CNN
        remaining = cntval;
        for ch in plain_chars_in_data:
            if(ch not in meta.label_dict):
                continue;
            new=this.grab_cluster(ch);
            ns=trchs.union(new);
            related_chars_in_data=related_chars_in_data.union(new);
            delta=len(ns)-len(trchs);
            if(delta<=remaining):
                trchs=ns;
                remaining-=delta;
        remaining=this.max_batch_size-meta.sp_cnt-len(trchs);
        plain_charid_not_in_data=list(meta.shaped_ids-related_chars_in_data);
        random.shuffle(plain_charid_not_in_data);
        for chid in plain_charid_not_in_data:
            if chid not in trchs:
                if (remaining == 0):
                    break;
                if (this.neg_servant==False and meta.masters[chid]!=chid):
                    continue;
                remaining-=1;
                trchs.add(chid);

        trsps=set([meta.label_dict[i] for i in meta.sp_tokens]);
        return trsps,trchs;

    def __init__(this,sampler_args):
        this.setup_sampler(sampler_args);
    def sample_charset_by_text(this,meta,text_batch,use_sp=True,device="cpu"):
        plain_chars_in_data = meta.get_occured(text_batch)
        trsps,trchs=this.get_sampled_ids(meta,plain_chars_in_data);
        trchs=list(trchs);
        if(use_sp is not False):
            trsps=list(trsps);
        else:
            trsps=[];
        plabels,gplabels,tdicts,gtdicts=meta.get_plabel_and_dictg(trsps,trchs,device=device)
        normprotos=[meta.norm_protos[i-meta.sp_cnt] for i in trchs];
        # this.debug(trchs,"meow");
        return normprotos,plabels,gplabels,tdicts,gtdicts;
