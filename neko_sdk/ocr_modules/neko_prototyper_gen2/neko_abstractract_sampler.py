import torch
from torch import nn
import numpy as np
from neko_2020nocr.dan.utils import decode_raw_static,\
    decode_prob_static,encode_static,decode_k_static,\
    decode_word_topk_static,decode_char_topk_static
from neko_sdk.ocr_modules.sptokens import tUNK,hUNK,tDC
from neko_sdk.ocr_modules.io.encdec import encode_fn_naive,encode_fn_naive_noeos;

# this class defines how samples are sampled ^_^


import regex
import copy
class neko_prototype_sampler_static(nn.Module):
    SPLIT=r"\X";
    def set_gbidict(this,oks):
        this.gbidict = {}
        gkset = {};
        gkcnt=0
        for k in oks:
            drk = this.label_dict[k];
            if (drk not in gkset):
                gkset[drk] = gkcnt;
                gkcnt += 1;
            this.label_dict[k] = gkset[drk];
            this.gbidict[k] = gkset[drk];
            if (drk == this.label_dict[k]):
                this.gbidict[gkset[drk]] = k;
            else:
                print("err");
                exit(9);

    def load_meta(this, meta):
        # Generally, CUDA devices works well with dynamic batch size.
        # However, for Rocm devices it bites as it compiles kernel
        # everytime we change batch size, it's nightmare.
        list_character = list(meta["chars"]);
        this.aligned_characters = meta["achars"];
        # characters without shape is generally what you do now want to sample.
        this.shaped_characters = set(meta["chars"])
        # UNK is not a sp_token as it is centerless.
        this.character = list(meta["sp_tokens"]) + list_character;
        this.label_dict = meta["label_dict"];

        this.shaped_ids = set([this.label_dict[i] for i in this.shaped_characters]);

        this.sp_cnt = len(meta["sp_tokens"]);
        this.sp_tokens = meta["sp_tokens"];
        this.norm_protos = meta["protos"][this.sp_cnt:];

        unk = this.label_dict[tUNK];
        # if the dict does not provide an specific unk token, set it to -1;
        for i, char in enumerate(this.character):
            # print(i, char)
            this.label_dict[char] = i;

        # shapeless unk shall be excluded
        if (unk < 0):
            this.label_set = set(this.label_dict.values()) - {unk};
        else:
            this.label_set = set(this.label_dict.values());

        for i in range(len(this.norm_protos)):
            if this.norm_protos[i] is not None and this.norm_protos[i].max() > 20:
                this.norm_protos[i] = (this.norm_protos[i] - 127.5) / 128;

        this.prototype_cnt = -1;
        # handles Capitalize like problem.
        # In some annotations they have same labels, while in others not.
        # i.e, 'A' and 'a' can have the same label 'a',
        # '茴','回','囘' and '囬' can have the same label '回'
        this.masters = meta["master"];
        this.reduced_label_dict = {}
        this.reduced_bidict = {}

        kcnt = 0;
        kset = {};
        ks = []
        ls = []
        for k in this.label_dict:
            ks.append(k);
            ls.append(this.label_dict[k]);
        oks = [ks[i] for i in np.argsort(ls)];

        for k in oks:
            if (this.label_dict[k] in this.masters):
                drk = this.masters[this.label_dict[k]];
            else:
                drk = this.label_dict[k];
            if (drk not in kset):
                kset[drk] = kcnt;
                kcnt += 1;
            this.reduced_label_dict[k] = kset[drk];
            this.reduced_bidict[k] = kset[drk];
            if (drk == this.label_dict[k]):
                this.reduced_bidict[kset[drk]] = k;

        this.set_gbidict(oks);

        # Foes includes the characters looks like each other
        # but never share labels (They may actually have linguistic relationships...
        # Like yanderes in a broken relationship[x]).
        # This set helps implement ohem like minibatching on the huge labelset.
        # e.g. 'u' and 'ü'
        this.foes = meta["foes"];
        this.servants = meta["servants"];
        # union set of friend, harem and foe.
        this.related_proto_ids = meta["relationships"];

    def setup_sampler(this, sampler_args):
        pass;

    def setup_meta(this, meta_args):
        this.EOS=0;
        this.case_sensitive = meta_args["case_sensitive"];
        this.meta_args=meta_args;
        this.masters_share = not this.case_sensitive;
        if(meta_args["meta_path"] is None):
            return ;
        meta = torch.load(meta_args["meta_path"]);

        this.load_meta(meta);


    def get_occured(this, text_batch):
        b = "";
        for _ in text_batch: b += _;
        return list(set(regex.findall(this.SPLIT, b, regex.U)));

    def dump_all_impl(this, use_sp=True):
        if (use_sp):
            trsps = list(set([this.label_dict[i] for i in this.sp_tokens]));
        else:
            trsps = [];
        trchs = list(set([this.label_dict[i] for i in this.shaped_characters]));
        normprotos = [this.norm_protos[i - this.sp_cnt] for i in trchs];
        plabels, tdicts = this.get_plabel_and_dict(trsps, trchs)
        return normprotos, plabels, tdicts;
    # that means every thing is sampled
    def dump_all(this, metaargs=None, use_sp=True):
        if (metaargs is not None):
            if (metaargs["meta_path"] is None):
                return None, None, None;
            a = copy.deepcopy(this);
            a.setup_meta(metaargs);
            return a.dump_all_impl(use_sp);
        return this.dump_all_impl(use_sp)
    # that means every thing is sampled.
    def dump_allg_impl(this, use_sp=True):
        if (use_sp):
            trsps = list(set([this.label_dict[i] for i in this.sp_tokens]));
        else:
            trsps = [];
        trchs = list(set([this.label_dict[i] for i in this.shaped_characters]));
        normprotos = [this.norm_protos[i - this.sp_cnt] for i in trchs];
        plabels, gplabels, bidict, gbidict = this.get_plabel_and_dictg(trsps, trchs)
        return normprotos, plabels, gplabels, bidict, gbidict;

    def dump_allg(this, metaargs=None, use_sp=True):
        if (metaargs is not None):
            a = copy.deepcopy(this);
            a.setup_meta(metaargs);
            return a.dump_allg_impl(use_sp);
        return this.dump_allg_impl(use_sp);


    def __init__(this, meta_args,sampler_args):
        super(neko_prototype_sampler_static, this).__init__()
        this.setup_meta(meta_args)
        this.setup_sampler(sampler_args);

    # No semb shit here, semb comes form meta, not sampler
    def get_plabel_and_dict_core(this, sappids, normpids, masters_share,device="cpu"):
        all_ids = sappids + normpids;
        new_id = 0;
        plabels = [];
        labmap = {};
        bidict = {}
        gbidict={}
        for i in all_ids:
            cha = this.aligned_characters[i];
            if (masters_share):
                vlab = this.masters[i];
            else:
                vlab = i;
            if (vlab not in labmap):
                labmap[vlab] = new_id;
                # A new label
                new_id += 1;
                # sembs.append(this.semantic_embedding[vlab]);
            alab = labmap[vlab];
            vcha=this.aligned_characters[vlab];
            plabels.append(alab);
            bidict[alab] = vcha;
            bidict[cha] = alab;

        plabels.append(new_id)
        bidict["[UNK]"] = new_id;

        # Well it has to be something --- at least ⑨ is alignment friendly and not likely to appear in the dataset
        bidict[new_id] = "⑨";
        return torch.tensor(plabels,device=device), bidict;

    # No semb shit here, semb comes form meta, not sampler
    def get_gplabel_and_dict_core(this, sappids, normpids, masters_share,use_sp=True,device="cpu"):
        if(use_sp):
            all_ids = sappids + normpids;
        else:
            all_ids=normpids;
        new_id = 0;
        plabels = [];
        labmap = {};
        bidict = {}
        gplabels=[];
        gmapping=[];
        for i in all_ids:
            cha = this.aligned_characters[i];
            if (masters_share):
                vlab = this.masters[i];
            else:
                vlab = i;
            vcha = this.aligned_characters[vlab];
            if (vlab not in labmap):
                labmap[vlab] = new_id;
                # A new label
                new_id += 1;
                # sembs.append(this.semantic_embedding[vlab]);
            alab = labmap[vlab];
            plabels.append(alab);
            bidict[alab] = vcha;
            bidict[cha] = alab;
        plabels.append(new_id)
        bidict["[UNK]"] = new_id;

        if(this.masters_share):
            gbidict=this.reduced_bidict
        else:
            gbidict=this.gbidict

        for i in range(new_id):
            gplabels.append(gbidict[bidict[i]]);
        gplabels.append(gbidict["[UNK]"]);
        gbidict[gbidict["[UNK]"]]="";
        # Well it has to be something --- at least ⑨ is alignment friendly and not likely to appear in the dataset
        # set most special keys to "" if any.
        for s in sappids:
            bidict[s]="";
        bidict[new_id] = "⑨";

        return torch.tensor(plabels,device=device),torch.tensor(gplabels,device=device), bidict,gbidict;

    def get_plabel_and_dict(this,sappids,normpids,device="cpu"):
        return this.get_plabel_and_dict_core(sappids,normpids,this.masters_share,device=device);
    def get_plabel_and_dictg(this,sappids,normpids,device="cpu"):
        return this.get_gplabel_and_dict_core(sappids,normpids,this.masters_share,device=device);

#################### These APIs will not be used in the NG framework.
#################### Encoding and decoding will be directly handled by the framework itself via tdict.
    def encode_noeos(this, proto, plabel, tdict, label_batch,device="cpu"):
        # inserting don't care.
        tdict[tDC]=-1;
        tdict[-1]=tDC;
        return encode_fn_naive_noeos(tdict, label_batch,this.case_sensitive,this.SPLIT,device=device)

    def encode(this, proto, plabel, tdict, label_batch,device="cpu"):
        if (not this.case_sensitive):
            label_batch = [l.lower() for l in label_batch]
        return encode_fn_naive(tdict, label_batch,this.case_sensitive,this.EOS,this.SPLIT,device=device)

    def decode(this, net_out, length, protos, labels, tdict):
        return decode_raw_static(net_out, length, tdict);
        # decoding prediction into text with geometric-mean probability
        # the probability is used to select the more realiable prediction when using bi-directional decoders
    def decode_beam(this, net_out, length, protos, labels, tdict,topk_ch=5,topk_beams=10):
        return decode_word_topk_static(net_out, length, protos, labels, tdict, topk_ch=topk_ch,topk_beams=topk_beams);

    def decode_beam_char(this, net_out, length, protos, labels, tdict,topk_ch=5):
        return decode_char_topk_static(net_out,length, protos, labels, tdict,topk_ch=topk_ch);


