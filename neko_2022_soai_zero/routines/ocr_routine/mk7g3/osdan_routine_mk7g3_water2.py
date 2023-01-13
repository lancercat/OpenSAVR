
from neko_2021_mjt.routines.subroutines.cores.mk7g3 import neko_HDOS2C_routine_CFmk7g3_core,recog_loss
from neko_2020nocr.dan.common.common import flatten_label
from neko_2021_mjt.routines.ocr_routines.mk7g3.osdan_routine_mk7g3 import neko_HDOS2C_routine_CFmk7g3

from neko_2021_mjt.modulars.neko_inflater import neko_inflater
import torch
from torch.nn import functional as trnf

class neko_cwag2_subroutine:
    pass;

class neko_cwag4ks_subroutine:
    def __init__(this,cnt=64,wprecon=1,withfrecon=True,domx_shuf=False,
                 domx_proto=True,domx_proto_detach=False,domx_feat=True,domx_feat_detach=False,featnorml=0.,cycshuf=True,
                 use_gt_in_cyc=False,det_proto_for_rec=False,cycknown=False,detfeforrec=True,cycfeat=False):
        this.CNT=cnt;
        this.wprecon=wprecon;
        this.withfrecon=withfrecon;
        this.domx_shuf=domx_shuf;
        this.domx_proto=domx_proto;
        this.domx_feat=domx_feat;
        this.domx_proto_detach=domx_proto_detach;
        this.domx_feat_detach=domx_feat_detach;
        this.featnorml=featnorml;
        this.cycshuf=cycshuf;
        this.use_gt_in_cyc=use_gt_in_cyc;
        this.det_proto_for_rec=det_proto_for_rec;
        this.cycknown=cycknown;
        this.cycfeat=cycfeat; # now we leave it out....




    def get_chunk(this,rec_trunks,mylength):
        if(len(rec_trunks)==0):
            return [0,mylength];
        else:
            return [rec_trunks[-1][-1],rec_trunks[-1][-1]+mylength];


    def fp_impl(this,fout_emb,proto_,normprotos_,plabels,label_flatten,device,modular_dict,tdict=None):
        if(this.featnorml<0.000009):
            loss=torch.tensor(0,device=device).float();
            terms = {};
        else:
            loss=trnf.relu(fout_emb.norm(p=2,dim=1)-3,inplace=False).mean()*this.featnorml;
            terms = {"fnorm":loss.item()};

        plabels_cu=plabels.to(proto_.device);
        cyc_ims = [];
        cyc_labels = [];
        psel=torch.randperm(proto_.shape[0])[:this.CNT].to(proto_.device);
        proto=proto_[psel];
        cls_proto=[proto_];
        cls_plabel=[plabels[:-1].to(proto_.device)];
        normprotos_cu=torch.cat(normprotos_).to(proto_.device);
        normprotos=normprotos_cu[psel];
        normprotos_label = plabels_cu[psel];


        shufed_ids = torch.randperm(fout_emb.shape[0],device=fout_emb.device);
        flab_shufed = label_flatten[shufed_ids];
        fsel_id = shufed_ids[flab_shufed != plabels_cu[-1]];
        fsel_id = torch.sort(fsel_id[:this.CNT].to(proto_.device))[0]
        normed_fout = trnf.normalize(fout_emb[fsel_id], dim=-1, p=2);
        flab = label_flatten[fsel_id].to(proto_.device);

        if(modular_dict["p_recon"]!="NEP_skipped_NEP"):
            rec_list = [];
            rec_label=[];
            rec_dict={};
            rec_trunks=[];

            if(modular_dict["p_recon_loss"]!="NEP_skipped_NEP"):
                rec_list.append(proto)
                rec_trunks.append(this.get_chunk(rec_trunks,proto.shape[0]));
                rec_label.append(normprotos_label);
                rec_dict["proto"]=len(rec_dict);
                if (this.use_gt_in_cyc):
                    cyc_ims.append(normprotos);
                    cyc_labels.append(normprotos_label);

            if (modular_dict["recon_char_pred"] != "NEP_skipped_NEP" or modular_dict["fpm_recon_loss"]!= "NEP_skipped_NEP"):
                if(this.cycfeat and modular_dict["f_recon"]=="NEP_skipped_NEP"):
                    rec_label.append(flab);
                    rec_dict["feat"]=len(rec_list);
                    rec_trunks.append(this.get_chunk(rec_trunks,normed_fout.shape[0]));
                    rec_list.append(normed_fout);


            if (modular_dict["shuf_img"] != "NEP_skipped_NEP"):
                with torch.no_grad():
                    shufp, mapping = modular_dict["shuf_img"](normprotos);
                pshufp = modular_dict["shuf_proto"]([shufp], use_sp=False);
                if(this.cycshuf):
                    labs = torch.range(0, len(shufp) - 1, device=shufp.device, dtype=plabels.dtype) + plabels[-1].item();
                    cls_plabel.append(labs)
                    cls_proto.append(pshufp);
                    rec_label.append(labs);
                    if (this.use_gt_in_cyc):
                        cyc_ims.append(shufp);
                        cyc_labels.append(labs);
                else:
                    # simply ignore them.
                    labs=torch.zeros(len(shufp), device=shufp.device, dtype=plabels.dtype)-1;
                    rec_label.append(torch.zeros_like(labs)-1)
                rec_dict["shuf"] = len(rec_list);
                rec_trunks.append(this.get_chunk(rec_trunks,pshufp.shape[0]));
                rec_list.append(pshufp);
            cft=torch.cat(rec_list);
            recons = modular_dict["p_recon"](cft.unsqueeze(-1).unsqueeze(-1));

            if (this.wprecon > 0 and modular_dict["p_recon_loss"]!="NEP_skipped_NEP"):
                l_precon = modular_dict["p_recon_loss"](recons[rec_trunks[rec_dict["proto"]][0]:rec_trunks[rec_dict["proto"]][1]], normprotos);
                loss += l_precon;
                terms["p_recon"] = l_precon.item();

            if (modular_dict["shuf_img"] != "NEP_skipped_NEP"):
                l_precon_shuf = modular_dict["shuf_recon_loss"](recons[rec_trunks[rec_dict["shuf"]][0]:rec_trunks[rec_dict["shuf"]][1]], shufp);
                loss += l_precon_shuf;
                terms["shuf_recon"] = l_precon_shuf.item();
            if (modular_dict["shuf_part_recon_loss"] != "NEP_skipped_NEP"):
                l_shufrec = modular_dict["shuf_part_recon_loss"](shufp, pshufp);
                loss += l_shufrec;
                terms["shuf_part_rec"] = l_shufrec.item();
                # A soft link to p_recon can be used here.
            if (modular_dict["fpm_recon_loss"] != "NEP_skipped_NEP" ):
                l_fpm_recon_loss=modular_dict["fpm_recon_loss"](recons[rec_trunks[rec_dict["feat"]][0]:rec_trunks[rec_dict["feat"]][1]],flab,normprotos_cu,plabels[:-1]);
                # l_fpm_recon_loss=modular_dict["fpm_recon_loss"](recons[rec_trunks[rec_dict["feat"]][0]:rec_trunks[rec_dict["feat"]][1]],flab,normprotos_cu,plabels[:-1],dbgkey="meow",tdict=tdict);
                loss += l_fpm_recon_loss;
                terms["fpm"] = l_fpm_recon_loss.item();
                # A soft link to p_recon can be used here.


            if (modular_dict["recon_char_fe"] != "NEP_skipped_NEP"):
                lrec=len(recons)
                all_labels=torch.cat(cyc_labels+rec_label);
                cyc_ims.append(recons);
                lcyc=all_labels.shape[0]-lrec;
                cls_plabel_c=torch.cat(cls_plabel);
                pred_f_recon = modular_dict["recon_char_fe"](torch.cat(cyc_ims));
                if(lcyc):
                    locyc,lorec=pred_f_recon.split([lcyc,lrec])
                    p=torch.cat(cls_proto);
                    f_recon_logitc = modular_dict["recon_char_pred"](locyc,p, cls_plabel_c);
                    # protos do not learn from reconstructed, which might include weird embeddings
                    if (this.det_proto_for_rec):
                        f_recon_logitr = modular_dict["recon_char_pred"](lorec,p.detach(), cls_plabel_c);
                    else:
                        f_recon_logitr = modular_dict["recon_char_pred"](lorec, p, cls_plabel_c);
                    l_frecon, _ = modular_dict["f_recon_loss"](None, torch.cat([f_recon_logitc,f_recon_logitr]), all_labels);
                    loss += l_frecon;
                    terms["cyc"] = l_frecon.item();
                else:
                # hard wired
                    cyc_ims=torch.cat(cyc_ims);
                    if(this.cycknown):
                        idx=(all_labels>=0)
                        cyc_ims=cyc_ims[idx];
                        all_labels=all_labels[idx];
                    pred_f_recon = modular_dict["recon_char_fe"](cyc_ims);
                    if (not this.det_proto_for_rec):
                        f_recon_logit = modular_dict["recon_char_pred"](pred_f_recon, torch.cat(cls_proto), cls_plabel_c);
                    else:
                        f_recon_logit = modular_dict["recon_char_pred"](pred_f_recon, torch.cat(cls_proto).detach(), cls_plabel_c);

                    l_frecon, _ = modular_dict["f_recon_loss"](None, f_recon_logit, all_labels);
                    loss += l_frecon;
                    terms["cyc"] = l_frecon.item();
            # if(modular_dict["fpm_recon_loss"]!="NEP_skipped_NEP"):
            #     f_recon = recons[rec_trunks[rec_dict["feat"]][0]:rec_trunks[rec_dict["feat"]][1]];
            #     l_fpm = modular_dict["fpm_recon_loss"](f_recon, normprotos_cu, plabels[:-1], flab, dbgkey=None).mean();
            #     terms["fpm_recon"]=l_fpm.item();
            #     loss+=l_fpm;
            else:
                pass;
        if (modular_dict["dom_mix"] != "NEP_skipped_NEP"):

            dmmx_list=[];
            if(this.domx_proto):
                if(this.domx_proto_detach):
                    dmmx_list.append(proto.detach());
                else:
                    dmmx_list.append(proto);
            if(this.domx_feat):
                if(this.domx_feat_detach):
                    dmmx_list.append(normed_fout.detach());
                else:
                    dmmx_list.append(normed_fout);
            if(this.domx_shuf):
                dmmx_list.append(pshufp)
            # cftl = torch.cat(
            #     [torch.zeros(dmmx_list[i].shape[0], dtype=torch.int64, device=dmmx_list[i].device) + i for i in
            #      range(len(dmmx_list))]);
            dmxloss=modular_dict["dom_mix"](dmmx_list);
            loss += dmxloss;
            terms["dom_mix"] = dmxloss.item();

        return loss,terms;

class neko_HDOS2C_routine_CFmk7g3_rec_cyc_core(neko_HDOS2C_routine_CFmk7g3_core):
    def arm_submodules(this):
        this.inflater = neko_inflater();
        this.water_mod=neko_cwag2_subroutine();



    def fp_impl(this, input_dict, exdict, modular_dict, logger_dict, device):
        clips = input_dict["image"];

        # Prototypes(sampled)
        # And this helps using SYNTH words in LSCT
        target = exdict["target"];
        length = exdict["length"];
        tdict = exdict["tdict"];
        normprotos = exdict["proto"];
        # semb=exdict["semb"];
        plabel = exdict["plabel"];

        prototyper = modular_dict["prototyper"]

        proto = prototyper(normprotos, use_sp=False);
        label_flatten, length = flatten_label(target, EOSlen=0, length=length);
        target, label_flatten, culength = target.to(device), label_flatten.to(device), length.long().to(device)
        out_emb, A, pred_length = this.fe_seq(clips.to(device), modular_dict, length);
        fout_emb, _ = this.inflater.inflate(out_emb, length);




        cls_loss, cls_terms, beams = recog_loss(modular_dict, pred_length, culength, fout_emb, proto, plabel,
                                                label_flatten, length, tdict);


        water_loss, water_term = this.water_mod.fp_impl(fout_emb, proto, normprotos,plabel, label_flatten, device,
                                                        modular_dict,tdict=tdict);


        loss = cls_loss + 0.1 * water_loss;
        # computing accuracy and loss
        tarswunk = ["".join([tdict[i.item()] for i in target[j]]).replace('[s]', "") for j in
                    range(len(target))];

        logger_dict["accr"].add_iter(beams[0], length, tarswunk)
        logger_dict["loss"].add_iter(loss, {"cls": cls_terms, "water": water_term})
        return loss;

class neko_HDOS2C_routine_CFmk7g3_rec_cyc3ks_core(neko_HDOS2C_routine_CFmk7g3_rec_cyc_core):
    def arm_submodules(this):
        this.inflater = neko_inflater();
        this.water_mod=neko_cwag4ks_subroutine();

class neko_HDOS2C_routine_CFmk7g3_rec_cyc3ks(neko_HDOS2C_routine_CFmk7g3):
    def set_etc(this, args):
        this.maxT = args["maxT"];
        this.core=neko_HDOS2C_routine_CFmk7g3_rec_cyc3ks_core();
class neko_HDOS2C_routine_CFmk7g3_cyc3ksanfr_core(neko_HDOS2C_routine_CFmk7g3_rec_cyc_core):
    def arm_submodules(this):
        this.inflater = neko_inflater();
        this.water_mod=neko_cwag4ks_subroutine(cycfeat=False,cycshuf=False);
class neko_HDOS2C_routine_CFmk7g3_cyc3ksanfr(neko_HDOS2C_routine_CFmk7g3):
    def set_etc(this, args):
        this.maxT = args["maxT"];
        this.core=neko_HDOS2C_routine_CFmk7g3_cyc3ksanfr_core();

# import cv2
# import torch
# from torch.nn import functional as trnf
# import numpy as np;
#
# from neko_2021_mjt.modulars.neko_inflater import neko_inflater
# from neko_2020nocr.dan.common.common import flatten_label
# from neko_2021_mjt.routines.subroutines.cores.mk7g3 import neko_HDOS2C_routine_CFmk7g3_core,recog_loss
# from neko_2021_mjt.routines.ocr_routines.mk7g3.osdan_routine_mk7g3 import neko_HDOS2C_routine_CFmk7g3
# from neko_2022_soai_zero.routines.ocr_routine.mk7g3.osdan_routine_mk7g3_rec_adapt import neko_HDOS2C_routine_CFmk7g3_fix_rec_anc_mask_core,neko_HDOS2C_routine_CFmk7g3_rec_anc_mask_fix
#
# def show_recon(recons,path=None):
#     im=(trnf.sigmoid(recons.squeeze(1))*255).detach().cpu().numpy().reshape(-1,32).astype(np.uint8);
#     if(path is not None):
#         cv2.imwrite(path,im);
#     else:
#         cv2.imshow("meow",im);
#         cv2.waitKey(0);
#
# class neko_cwag2_subroutine:
#
#     def fp_impl(this,fout_emb,proto,normprotos,plabels,label_flatten,device,modular_dict,tdict=None):
#         loss=torch.tensor(0,device=device).float();
#         normed_fout=trnf.normalize(fout_emb,dim=-1,p=2);
#
#         terms={};
#         if(modular_dict["dom_mix"]!="NEP_skipped_NEP"):
#             l_dommix=modular_dict["dom_mix"]([proto,normed_fout]);
#             loss += l_dommix;
#             terms["dom_mix"] = l_dommix.item();
#
#         if(modular_dict["p_recon"]!="NEP_skipped_NEP"):
#             p_recon=modular_dict["p_recon"](proto.unsqueeze(-1).unsqueeze(-1));
#             l_precon=modular_dict["p_recon_loss"](p_recon,torch.cat(normprotos));
#             loss+=l_precon;
#             terms["p_recon"]=l_precon.item();
#         if(modular_dict["f_recon"]!="NEP_skipped_NEP"):
#             f_recon=modular_dict["f_recon"](normed_fout[:64].unsqueeze(-1).unsqueeze(-1));
#             if (modular_dict["p_recon"] != "NEP_skipped_NEP"):
#                 recon=torch.cat([p_recon,f_recon]);
#                 recon_label=torch.cat([plabels[:p_recon.shape[0]].to(p_recon.device),label_flatten[:f_recon.shape[0]]]);
#             else:
#                 recon=f_recon;
#                 recon_label=label_flatten[:64]
#             recon_pred=modular_dict["recon_char_fe"](recon);
#             recon_logit=modular_dict["recon_char_pred"](recon_pred,proto,plabels);
#             l_frecon,_=modular_dict["f_recon_loss"](None,recon_logit,recon_label);
#             loss += l_frecon;
#             terms["cycloss"] = l_frecon.item();
#         return loss,terms;
#
# class neko_HDOS2C_routine_CFmk7g3_rec_cyc_core(neko_HDOS2C_routine_CFmk7g3_core):
#     def arm_submodules(this):
#         this.inflater = neko_inflater();
#         this.water_mod=neko_cwag2_subroutine();
#
#
#
#     def fp_impl(this, input_dict, exdict, modular_dict, logger_dict, device):
#         clips = input_dict["image"];
#
#         # Prototypes(sampled)
#         # And this helps using SYNTH words in LSCT
#         target = exdict["target"];
#         length = exdict["length"];
#         tdict = exdict["tdict"];
#         normprotos = exdict["proto"];
#         # semb=exdict["semb"];
#         plabel = exdict["plabel"];
#
#         prototyper = modular_dict["prototyper"]
#
#         proto = prototyper(normprotos, use_sp=False);
#         label_flatten, length = flatten_label(target, EOSlen=0, length=length);
#         target, label_flatten, culength = target.to(device), label_flatten.to(device), length.long().to(device)
#         out_emb, A, pred_length = this.fe_seq(clips.to(device), modular_dict, length);
#         fout_emb, _ = this.inflater.inflate(out_emb, length);
#
#
#
#
#         cls_loss, cls_terms, beams = recog_loss(modular_dict, pred_length, culength, fout_emb, proto, plabel,
#                                                 label_flatten, length, tdict);
#
#
#         water_loss, water_term = this.water_mod.fp_impl(fout_emb, proto, normprotos,plabel, label_flatten, device,
#                                                         modular_dict,tdict=tdict);
#
#
#         loss = cls_loss + 0.1 * water_loss;
#         # computing accuracy and loss
#         tarswunk = ["".join([tdict[i.item()] for i in target[j]]).replace('[s]', "") for j in
#                     range(len(target))];
#
#         logger_dict["accr"].add_iter(beams[0], length, tarswunk)
#         logger_dict["loss"].add_iter(loss, {"cls": cls_terms, "water": water_term})
#         return loss;
#
# class neko_HDOS2C_routine_CFmk7g3_rec_cyc(neko_HDOS2C_routine_CFmk7g3):
#     def set_etc(this, args):
#         this.maxT = args["maxT"];
#         this.core=neko_HDOS2C_routine_CFmk7g3_rec_cyc_core();
#
#
#
#
# class neko_cwag3_subroutine:
#     def debug(this,recons,tag="tmp"):
#         cv2.imshow(tag, (recons[-16:].reshape(16 * 32, 32).detach().cpu().numpy() * 64 + 127).astype(np.uint8));
#         cv2.waitKey(0);
#     def fp_impl(this,fout_emb,proto,normprotos,plabels,label_flatten,device,modular_dict,tdict=None):
#         loss=torch.tensor(0,device=device).float();
#         normed_fout=trnf.normalize(fout_emb,dim=-1,p=2);
#
#         terms={};
#         if(modular_dict["dom_mix"]!="NEP_skipped_NEP"):
#             l_dommix=modular_dict["dom_mix"]([proto,normed_fout]);
#             loss += l_dommix;
#             terms["dom_mix"] = l_dommix.item();
#
#         if(modular_dict["p_recon"]!="NEP_skipped_NEP"):
#             reconr=modular_dict["p_recon"](proto.unsqueeze(-1).unsqueeze(-1));
#             l_precon=modular_dict["p_recon_loss"](reconr,torch.cat(normprotos));
#             loss+=l_precon;
#             terms["p_recon"]=l_precon.item();
#
#
#         if(modular_dict["shuf_img"]!="NEP_skipped_NEP"):
#             shufp,mapping=modular_dict["shuf_img"](torch.cat(normprotos[:64]));
#             pshufp=modular_dict["prototyper"]([shufp],use_sp=False);
#             # A soft link to p_recon can be used here.
#             reconshuf=modular_dict["shuf_recon"](pshufp.unsqueeze(-1).unsqueeze(-1));
#             l_precon_shuf=modular_dict["shuf_recon_loss"](reconshuf,shufp);
#             loss+=l_precon_shuf;
#             terms["shuf_recon"]=l_precon_shuf.item();
#         # with torch.no_grad():
#         #     reconrd = modular_dict["p_recon"](fout_emb[:32].unsqueeze(-1).unsqueeze(-1));
#         #     this.debug(reconrd, "tmp");
#         #     this.debug(reconr, "tmp2");
#         #     this.debug(reconshuf,"tmp3");
#
#         if(modular_dict["f_recon"]!="NEP_skipped_NEP"):
#             f_recon=modular_dict["f_recon"](normed_fout[:64].unsqueeze(-1).unsqueeze(-1));
#             pcnt=min(reconr.shape[0],64);
#
#             recons=torch.cat([f_recon,reconr[:pcnt]]);
#             labels=torch.cat([label_flatten[:64],plabels[:pcnt].to(label_flatten.device)]);
#
#             pred_f_recon=modular_dict["recon_char_fe"](recons);
#             f_recon_logit=modular_dict["recon_char_pred"](pred_f_recon,proto,plabels);
#             l_frecon,_=modular_dict["f_recon_loss"](None,f_recon_logit,labels);
#             loss += l_frecon;
#             terms["f_recon"] = l_frecon.item();
#         return loss,terms;
#
#
# class neko_HDOS2C_routine_CFmk7g3_rec_cyc2_core(neko_HDOS2C_routine_CFmk7g3_rec_cyc_core):
#     def arm_submodules(this):
#         this.inflater = neko_inflater();
#         this.water_mod=neko_cwag3_subroutine();
#
# class neko_HDOS2C_routine_CFmk7g3_rec_cyc2(neko_HDOS2C_routine_CFmk7g3):
#     def set_etc(this, args):
#         this.maxT = args["maxT"];
#         this.core=neko_HDOS2C_routine_CFmk7g3_rec_cyc2_core();
#
#
# class neko_HDOS2C_routine_CFmk7g3_rec_anc_mask_fix_cyc2_core(neko_HDOS2C_routine_CFmk7g3_fix_rec_anc_mask_core):
#     def arm_submodules(this):
#         this.inflater = neko_inflater();
#         this.water_mod=neko_cwag3_subroutine();
#
# class neko_HDOS2C_routine_CFmk7g3_rec_anc_mask_fix_cyc2(neko_HDOS2C_routine_CFmk7g3):
#     def set_etc(this, args):
#         this.maxT = args["maxT"];
#         this.core=neko_HDOS2C_routine_CFmk7g3_rec_anc_mask_fix_cyc2_core();
#
#
#
# class neko_cwag4_subroutine:
#     def debug(this,recons,tag="tmp"):
#         cv2.imshow(tag, (recons[-16:].reshape(16 * 32, 32).detach().cpu().numpy() * 64 + 127).astype(np.uint8));
#         cv2.waitKey(0);
#     def fp_impl(this,fout_emb,proto,normprotos,plabels,label_flatten,device,modular_dict,tdict=None):
#         loss=torch.tensor(0,device=device).float();
#         normed_fout=trnf.normalize(fout_emb,dim=-1,p=2);
#
#         terms={};
#         if(modular_dict["dom_mix"]!="NEP_skipped_NEP"):
#             l_dommix=modular_dict["dom_mix"]([proto,normed_fout]);
#             loss += l_dommix;
#             terms["dom_mix"] = l_dommix.item();
#
#         if(modular_dict["shuf_img"]!="NEP_skipped_NEP"):
#             shufp,mapping=modular_dict["shuf_img"](torch.cat(normprotos[:64]));
#             pshufp=modular_dict["prototyper"]([shufp],use_sp=False);
#             # A soft link to p_recon can be used here.
#             reconshuf=modular_dict["shuf_recon"](pshufp.unsqueeze(-1).unsqueeze(-1));
#             l_precon_shuf=modular_dict["shuf_recon_loss"](reconshuf,shufp);
#             loss+=l_precon_shuf;
#             terms["shuf_recon"]=l_precon_shuf.item();
#
#         if(modular_dict["p_recon"]!="NEP_skipped_NEP"):
#             recons=modular_dict["f_recon"](torch.cat([proto,normed_fout]).unsqueeze(-1).unsqueeze(-1));
#             labels=torch.cat([plabels.to(label_flatten.device)[:proto.shape[0]],label_flatten]);
#             reconr=recons[:proto.shape[0]];
#             l_precon=modular_dict["p_recon_loss"](reconr,torch.cat(normprotos));
#             loss+=l_precon;
#             terms["p_recon"] = l_precon.item();
#             pred_f_recon=modular_dict["recon_char_fe"](recons);
#             # this.debug(recons,"tmp");
#             # this.debug(reconr,"tmp2");
#
#             f_recon_logit=modular_dict["recon_char_pred"](pred_f_recon,proto,plabels);
#             l_frecon,_=modular_dict["f_recon_loss"](None,f_recon_logit,labels);
#             loss += l_frecon;
#             terms["f_recon"] = l_frecon.item();
#         return loss,terms;
#
#
# class neko_HDOS2C_routine_CFmk7g3_rec_cyc3_core(neko_HDOS2C_routine_CFmk7g3_rec_cyc_core):
#     def arm_submodules(this):
#         this.inflater = neko_inflater();
#         this.water_mod=neko_cwag4_subroutine();
#
# class neko_HDOS2C_routine_CFmk7g3_rec_cyc3(neko_HDOS2C_routine_CFmk7g3):
#     def set_etc(this, args):
#         this.maxT = args["maxT"];
#         this.core=neko_HDOS2C_routine_CFmk7g3_rec_cyc3_core();
#
# class neko_cwag4k_subroutine:
#     def __init__(this,cnt=64,wprecon=1,withfrecon=True,detach_proto_mva=False,domx_shuf=False,domx_proto=True,domx_feat=True):
#         this.CNT=cnt;
#         this.wprecon=wprecon;
#         this.withfrecon=withfrecon;
#         this.mva_detach_proto=detach_proto_mva;
#         this.domx_shuf=domx_shuf;
#         this.domx_proto=domx_proto;
#         this.domx_feat=domx_feat;
#
#     def fp_impl(this,fout_emb,proto_,normprotos_,plabels,label_flatten,device,modular_dict):
#         loss=torch.tensor(0,device=device).float();
#         psel=torch.randperm(proto_.shape[0])[:this.CNT].to(proto_.device);
#         fsela=torch.randperm(fout_emb.shape[0]);
#         flaba=label_flatten[fsela];
#         fsel=fsela[flaba!=plabels[-1]];
#         fsel=fsel[:this.CNT].to(proto_.device)
#         normed_fout=trnf.normalize(fout_emb[fsel],dim=-1,p=2);
#         flab=label_flatten[fsel];
#
#         proto=proto_[psel];
#         normprotos_cu=torch.cat(normprotos_).to(proto_.device);
#         normprotos=normprotos_cu[psel];
#         terms={};
#
#         if(modular_dict["shuf_img"]!="NEP_skipped_NEP"):
#             shufp,mapping=modular_dict["shuf_img"](normprotos);
#             pshufp=modular_dict["shuf_proto"]([shufp],use_sp=False);
#             if(modular_dict["shuf_sim_loss"] != "NEP_skipped_NEP"):
#                 l_shufsim=modular_dict["shuf_sim_loss"](proto,pshufp,mapping);
#                 loss+=l_shufsim;
#                 terms["shuf_sim"] = l_shufsim.item();
#             if (modular_dict["shuf_part_recon_loss"] != "NEP_skipped_NEP"):
#                 l_shufrec = modular_dict["shuf_part_recon_loss"](shufp,pshufp);
#                 loss += l_shufrec;
#                 terms["shuf_part_rec"] = l_shufrec.item();
#             # A soft link to p_recon can be used here.
#             reconshuf=modular_dict["shuf_recon"](pshufp.unsqueeze(-1).unsqueeze(-1));
#             l_precon_shuf=modular_dict["shuf_recon_loss"](reconshuf,shufp);
#             loss+=l_precon_shuf;
#             terms["shuf_recon"]=l_precon_shuf.item();
#         if(modular_dict["fpm_recon_loss"]!="NEP_skipped_NEP"):
#             f_recon=modular_dict["f_recon"](normed_fout.unsqueeze(-1).unsqueeze(-1));
#             l_fpm=modular_dict["fpm_recon_loss"](f_recon,normprotos_cu,plabels[:-1],flab,dbgkey=None).mean();
#             terms["fpm_recon"]=l_fpm.item();
#             loss+=l_fpm;
#
#         if(modular_dict["dom_mix"]!="NEP_skipped_NEP"):
#             dta=[]; # domain targets
#             if(this.domx_feat==True):
#                 dta.append(normed_fout);
#             elif(this.domx_feat=="detached"):
#                 dta.append(normed_fout.detach());
#
#             if(this.domx_proto==True):
#                 dta.append(proto);
#             elif (this.domx_proto == "detached"):
#                 dta.append(proto.detach());
#
#             if(this.domx_shuf==True):
#                 dta.append(pshufp);
#
#             l_dommix=modular_dict["dom_mix"](dta);
#             loss += l_dommix;
#             terms["dom_mix"] = l_dommix.item();
#
#         if(modular_dict["p_recon"]!="NEP_skipped_NEP"):
#             if(this.withfrecon and  modular_dict["recon_char_fe"]!="NEP_skipped_NEP"):
#                 recons=modular_dict["p_recon"](torch.cat([proto,normed_fout]).unsqueeze(-1).unsqueeze(-1));
#                 labels=torch.cat([plabels[psel].to(label_flatten.device),label_flatten[fsel]]);
#                 pred_f_recon = modular_dict["recon_char_fe"](recons);
#                 f_recon_logit = modular_dict["recon_char_pred"](pred_f_recon, proto_, plabels);
#                 l_frecon, _ = modular_dict["f_recon_loss"](None, f_recon_logit, labels);
#                 loss += l_frecon;
#                 terms["f_recon"] = l_frecon.item();
#                 reconr=recons[:proto.shape[0]];
#             else:
#                 reconr=modular_dict["p_recon"](proto.unsqueeze(-1).unsqueeze(-1));
#             if(this.wprecon>0):
#                 l_precon=modular_dict["p_recon_loss"](reconr,normprotos);
#                 loss+=l_precon;
#                 terms["p_recon"] = l_precon.item();
#             else:
#                 pass;
#
#         return loss,terms;
#
# class neko_HDOS2C_routine_CFmk7g3_cyc3k_core(neko_HDOS2C_routine_CFmk7g3_rec_cyc_core):
#     def arm_submodules(this):
#         this.inflater = neko_inflater();
#         this.water_mod=neko_cwag4k_subroutine(wprecon=0);
#
#
# class neko_HDOS2C_routine_CFmk7g3_rec_core(neko_HDOS2C_routine_CFmk7g3_rec_cyc_core):
#     def arm_submodules(this):
#         this.inflater = neko_inflater();
#         this.water_mod=neko_cwag4k_subroutine(withfrecon=False);
#
# class neko_HDOS2C_routine_CFmk7g3_rec_cyc3k_core(neko_HDOS2C_routine_CFmk7g3_rec_cyc_core):
#     def arm_submodules(this):
#         this.inflater = neko_inflater();
#         this.water_mod=neko_cwag4k_subroutine();
# class neko_HDOS2C_routine_CFmk7g3_rec_cyc3k128_core(neko_HDOS2C_routine_CFmk7g3_rec_cyc_core):
#     def arm_submodules(this):
#         this.inflater = neko_inflater();
#         this.water_mod=neko_cwag4k_subroutine(128);
# class neko_HDOS2C_routine_CFmk7g3_cyc3k(neko_HDOS2C_routine_CFmk7g3):
#     def set_etc(this, args):
#         this.maxT = args["maxT"];
#         this.core=neko_HDOS2C_routine_CFmk7g3_cyc3k_core();
#
# class neko_HDOS2C_routine_CFmk7g3_rec_cyc3k(neko_HDOS2C_routine_CFmk7g3):
#     def set_etc(this, args):
#         this.maxT = args["maxT"];
#         this.core=neko_HDOS2C_routine_CFmk7g3_rec_cyc3k_core();
#
# class neko_HDOS2C_routine_CFmk7g3_rec(neko_HDOS2C_routine_CFmk7g3):
#     def set_etc(this, args):
#         this.maxT = args["maxT"];
#         this.core=neko_HDOS2C_routine_CFmk7g3_rec_core();
#
# class neko_HDOS2C_routine_CFmk7g3_rec_cyc3k128(neko_HDOS2C_routine_CFmk7g3):
#     def set_etc(this, args):
#         this.maxT = args["maxT"];
#         this.core=neko_HDOS2C_routine_CFmk7g3_rec_cyc3k128_core();
#
#
# class neko_HDOS2C_routine_CFmk7g3_cyc3k_dmxs_core(neko_HDOS2C_routine_CFmk7g3_rec_cyc_core):
#     def arm_submodules(this):
#         this.inflater = neko_inflater();
#         this.water_mod=neko_cwag4k_subroutine(wprecon=0,domx_shuf=True);
#
# class neko_HDOS2C_routine_CFmk7g3_cyc3k_dmxs(neko_HDOS2C_routine_CFmk7g3):
#     def set_etc(this, args):
#         this.maxT = args["maxT"];
#         this.core=neko_HDOS2C_routine_CFmk7g3_cyc3k_dmxs_core();
#
#
#
# class neko_cwag4ks_subroutine:
#     def __init__(this,cnt=64,wprecon=1,withfrecon=True,domx_shuf=False,
#                  domx_proto=True,domx_proto_detach=False,domx_feat=True,domx_feat_detach=False,featnorml=0.,cycshuf=True,
#                  use_gt_in_cyc=False,det_proto_for_rec=False,cycknown=False,detfeforrec=True,cycfeat=False):
#         this.CNT=cnt;
#         this.wprecon=wprecon;
#         this.withfrecon=withfrecon;
#         this.domx_shuf=domx_shuf;
#         this.domx_proto=domx_proto;
#         this.domx_feat=domx_feat;
#         this.domx_proto_detach=domx_proto_detach;
#         this.domx_feat_detach=domx_feat_detach;
#         this.featnorml=featnorml;
#         this.cycshuf=cycshuf;
#         this.use_gt_in_cyc=use_gt_in_cyc;
#         this.det_proto_for_rec=det_proto_for_rec;
#         this.cycknown=cycknown;
#         this.cycfeat=cycfeat; # now we leave it out....
#
#
#
#
#     def get_chunk(this,rec_trunks,mylength):
#         if(len(rec_trunks)==0):
#             return [0,mylength];
#         else:
#             return [rec_trunks[-1][-1],rec_trunks[-1][-1]+mylength];
#
#
#     def fp_impl(this,fout_emb,proto_,normprotos_,plabels,label_flatten,device,modular_dict,tdict=None):
#         if(this.featnorml<0.000009):
#             loss=torch.tensor(0,device=device).float();
#             terms = {};
#         else:
#             loss=trnf.relu(fout_emb.norm(p=2,dim=1)-3,inplace=False).mean()*this.featnorml;
#             terms = {"fnorm":loss.item()};
#
#         plabels_cu=plabels.to(proto_.device);
#         cyc_ims = [];
#         cyc_labels = [];
#         psel=torch.randperm(proto_.shape[0])[:this.CNT].to(proto_.device);
#         proto=proto_[psel];
#         cls_proto=[proto_];
#         cls_plabel=[plabels[:-1].to(proto_.device)];
#         normprotos_cu=torch.cat(normprotos_).to(proto_.device);
#         normprotos=normprotos_cu[psel];
#         normprotos_label = plabels_cu[psel];
#
#
#         shufed_ids = torch.randperm(fout_emb.shape[0],device=fout_emb.device);
#         flab_shufed = label_flatten[shufed_ids];
#         fsel_id = shufed_ids[flab_shufed != plabels_cu[-1]];
#         fsel_id = torch.sort(fsel_id[:this.CNT].to(proto_.device))[0]
#         normed_fout = trnf.normalize(fout_emb[fsel_id], dim=-1, p=2);
#         flab = label_flatten[fsel_id].to(proto_.device);
#
#         if(modular_dict["p_recon"]!="NEP_skipped_NEP"):
#             rec_list = [];
#             rec_label=[];
#             rec_dict={};
#             rec_trunks=[];
#
#             if(modular_dict["p_recon_loss"]!="NEP_skipped_NEP"):
#                 rec_list.append(proto)
#                 rec_trunks.append(this.get_chunk(rec_trunks,proto.shape[0]));
#                 rec_label.append(normprotos_label);
#                 rec_dict["proto"]=len(rec_dict);
#                 if (this.use_gt_in_cyc):
#                     cyc_ims.append(normprotos);
#                     cyc_labels.append(normprotos_label);
#
#             if (modular_dict["recon_char_pred"] != "NEP_skipped_NEP" or modular_dict["fpm_recon_loss"]!= "NEP_skipped_NEP"):
#                 if(this.cycfeat and modular_dict["f_recon"]=="NEP_skipped_NEP"):
#                     rec_label.append(flab);
#                     rec_dict["feat"]=len(rec_list);
#                     rec_trunks.append(this.get_chunk(rec_trunks,normed_fout.shape[0]));
#                     rec_list.append(normed_fout);
#
#
#             if (modular_dict["shuf_img"] != "NEP_skipped_NEP"):
#                 with torch.no_grad():
#                     shufp, mapping = modular_dict["shuf_img"](normprotos);
#                 pshufp = modular_dict["shuf_proto"]([shufp], use_sp=False);
#                 if(this.cycshuf):
#                     labs = torch.range(0, len(shufp) - 1, device=shufp.device, dtype=plabels.dtype) + plabels[-1].item();
#                     cls_plabel.append(labs)
#                     cls_proto.append(pshufp);
#                     rec_label.append(labs);
#                     if (this.use_gt_in_cyc):
#                         cyc_ims.append(shufp);
#                         cyc_labels.append(labs);
#                 else:
#                     # simply ignore them.
#                     labs=torch.zeros(len(shufp), device=shufp.device, dtype=plabels.dtype)-1;
#                     rec_label.append(torch.zeros_like(labs)-1)
#                 rec_dict["shuf"] = len(rec_list);
#                 rec_trunks.append(this.get_chunk(rec_trunks,pshufp.shape[0]));
#                 rec_list.append(pshufp);
#             cft=torch.cat(rec_list);
#             recons = modular_dict["p_recon"](cft.unsqueeze(-1).unsqueeze(-1));
#
#             if (this.wprecon > 0 and modular_dict["p_recon_loss"]!="NEP_skipped_NEP"):
#                 l_precon = modular_dict["p_recon_loss"](recons[rec_trunks[rec_dict["proto"]][0]:rec_trunks[rec_dict["proto"]][1]], normprotos);
#                 loss += l_precon;
#                 terms["p_recon"] = l_precon.item();
#
#             if (modular_dict["shuf_img"] != "NEP_skipped_NEP"):
#                 l_precon_shuf = modular_dict["shuf_recon_loss"](recons[rec_trunks[rec_dict["shuf"]][0]:rec_trunks[rec_dict["shuf"]][1]], shufp);
#                 loss += l_precon_shuf;
#                 terms["shuf_recon"] = l_precon_shuf.item();
#             if (modular_dict["shuf_part_recon_loss"] != "NEP_skipped_NEP"):
#                 l_shufrec = modular_dict["shuf_part_recon_loss"](shufp, pshufp);
#                 loss += l_shufrec;
#                 terms["shuf_part_rec"] = l_shufrec.item();
#                 # A soft link to p_recon can be used here.
#             if (modular_dict["fpm_recon_loss"] != "NEP_skipped_NEP" ):
#                 l_fpm_recon_loss=modular_dict["fpm_recon_loss"](recons[rec_trunks[rec_dict["feat"]][0]:rec_trunks[rec_dict["feat"]][1]],flab,normprotos_cu,plabels[:-1]);
#                 # l_fpm_recon_loss=modular_dict["fpm_recon_loss"](recons[rec_trunks[rec_dict["feat"]][0]:rec_trunks[rec_dict["feat"]][1]],flab,normprotos_cu,plabels[:-1],dbgkey="meow",tdict=tdict);
#                 loss += l_fpm_recon_loss;
#                 terms["fpm"] = l_fpm_recon_loss.item();
#                 # A soft link to p_recon can be used here.
#
#
#             if (modular_dict["recon_char_fe"] != "NEP_skipped_NEP"):
#                 lrec=len(recons)
#                 all_labels=torch.cat(cyc_labels+rec_label);
#                 cyc_ims.append(recons);
#                 lcyc=all_labels.shape[0]-lrec;
#                 cls_plabel_c=torch.cat(cls_plabel);
#                 pred_f_recon = modular_dict["recon_char_fe"](torch.cat(cyc_ims));
#                 if(lcyc):
#                     locyc,lorec=pred_f_recon.split([lcyc,lrec])
#                     p=torch.cat(cls_proto);
#                     f_recon_logitc = modular_dict["recon_char_pred"](locyc,p, cls_plabel_c);
#                     # protos do not learn from reconstructed, which might include weird embeddings
#                     if (this.det_proto_for_rec):
#                         f_recon_logitr = modular_dict["recon_char_pred"](lorec,p.detach(), cls_plabel_c);
#                     else:
#                         f_recon_logitr = modular_dict["recon_char_pred"](lorec, p, cls_plabel_c);
#                     l_frecon, _ = modular_dict["f_recon_loss"](None, torch.cat([f_recon_logitc,f_recon_logitr]), all_labels);
#                     loss += l_frecon;
#                     terms["cyc"] = l_frecon.item();
#                 else:
#                 # hard wired
#                     cyc_ims=torch.cat(cyc_ims);
#                     if(this.cycknown):
#                         idx=(all_labels>=0)
#                         cyc_ims=cyc_ims[idx];
#                         all_labels=all_labels[idx];
#                     pred_f_recon = modular_dict["recon_char_fe"](cyc_ims);
#                     if (not this.det_proto_for_rec):
#                         f_recon_logit = modular_dict["recon_char_pred"](pred_f_recon, torch.cat(cls_proto), cls_plabel_c);
#                     else:
#                         f_recon_logit = modular_dict["recon_char_pred"](pred_f_recon, torch.cat(cls_proto).detach(), cls_plabel_c);
#
#                     l_frecon, _ = modular_dict["f_recon_loss"](None, f_recon_logit, all_labels);
#                     loss += l_frecon;
#                     terms["cyc"] = l_frecon.item();
#             # if(modular_dict["fpm_recon_loss"]!="NEP_skipped_NEP"):
#             #     f_recon = recons[rec_trunks[rec_dict["feat"]][0]:rec_trunks[rec_dict["feat"]][1]];
#             #     l_fpm = modular_dict["fpm_recon_loss"](f_recon, normprotos_cu, plabels[:-1], flab, dbgkey=None).mean();
#             #     terms["fpm_recon"]=l_fpm.item();
#             #     loss+=l_fpm;
#             else:
#                 pass;
#         if (modular_dict["dom_mix"] != "NEP_skipped_NEP"):
#
#             dmmx_list=[];
#             if(this.domx_proto):
#                 if(this.domx_proto_detach):
#                     dmmx_list.append(proto.detach());
#                 else:
#                     dmmx_list.append(proto);
#             if(this.domx_feat):
#                 if(this.domx_feat_detach):
#                     dmmx_list.append(normed_fout.detach());
#                 else:
#                     dmmx_list.append(normed_fout);
#             if(this.domx_shuf):
#                 dmmx_list.append(pshufp)
#             # cftl = torch.cat(
#             #     [torch.zeros(dmmx_list[i].shape[0], dtype=torch.int64, device=dmmx_list[i].device) + i for i in
#             #      range(len(dmmx_list))]);
#             dmxloss=modular_dict["dom_mix"](dmmx_list);
#             loss += dmxloss;
#             terms["dom_mix"] = dmxloss.item();
#
#         return loss,terms;
#
# class neko_HDOS2C_routine_CFmk7g3_cyc3ks_dmxdp_core(neko_HDOS2C_routine_CFmk7g3_rec_cyc_core):
#     def arm_submodules(this):
#         this.inflater = neko_inflater();
#         this.water_mod=neko_cwag4ks_subroutine(domx_proto_detach=True);
#
# class neko_HDOS2C_routine_CFmk7g3_cyc3ks_dmxdp(neko_HDOS2C_routine_CFmk7g3):
#     def set_etc(this, args):
#         this.maxT = args["maxT"];
#         this.core=neko_HDOS2C_routine_CFmk7g3_cyc3ks_dmxdp_core();
#
#
#
# class neko_HDOS2C_routine_CFmk7g3_cyc3ks_dmxs_core(neko_HDOS2C_routine_CFmk7g3_rec_cyc_core):
#     def arm_submodules(this):
#         this.inflater = neko_inflater();
#         this.water_mod=neko_cwag4ks_subroutine(domx_shuf=True);
#
# class neko_HDOS2C_routine_CFmk7g3_cyc3ks_dmxs(neko_HDOS2C_routine_CFmk7g3):
#     def set_etc(this, args):
#         this.maxT = args["maxT"];
#         this.core=neko_HDOS2C_routine_CFmk7g3_cyc3ks_dmxs_core();
#
#
#
# class neko_HDOS2C_routine_CFmk7g3_cyc3ksnfr_core(neko_HDOS2C_routine_CFmk7g3_rec_cyc_core):
#     def arm_submodules(this):
#         this.inflater = neko_inflater();
#         this.water_mod=neko_cwag4ks_subroutine(cycfeat=False);
#
#
# class neko_HDOS2C_routine_CFmk7g3_cyc3ksnfr(neko_HDOS2C_routine_CFmk7g3):
#     def set_etc(this, args):
#         this.maxT = args["maxT"];
#         this.core=neko_HDOS2C_routine_CFmk7g3_cyc3ksnfr_core();
#
# class neko_HDOS2C_routine_CFmk7g3_cyc3ksanfr_core(neko_HDOS2C_routine_CFmk7g3_rec_cyc_core):
#     def arm_submodules(this):
#         this.inflater = neko_inflater();
#         this.water_mod=neko_cwag4ks_subroutine(cycfeat=False,cycshuf=False);
# class neko_HDOS2C_routine_CFmk7g3_cyc3ksanfr(neko_HDOS2C_routine_CFmk7g3):
#     def set_etc(this, args):
#         this.maxT = args["maxT"];
#         this.core=neko_HDOS2C_routine_CFmk7g3_cyc3ksanfr_core();
#
# class neko_HDOS2C_routine_CFmk7g3_cyc3ksagnfr_core(neko_HDOS2C_routine_CFmk7g3_rec_cyc_core):
#     def arm_submodules(this):
#         this.inflater = neko_inflater();
#         this.water_mod=neko_cwag4ks_subroutine(cycfeat=False,cycshuf=False,use_gt_in_cyc=True);
# class neko_HDOS2C_routine_CFmk7g3_cyc3ksagnfr(neko_HDOS2C_routine_CFmk7g3):
#     def set_etc(this, args):
#         this.maxT = args["maxT"];
#         this.core=neko_HDOS2C_routine_CFmk7g3_cyc3ksagnfr_core();
#
# class neko_HDOS2C_routine_CFmk7g3_cyc3ksgnfr_core(neko_HDOS2C_routine_CFmk7g3_rec_cyc_core):
#     def arm_submodules(this):
#         this.inflater = neko_inflater();
#         this.water_mod=neko_cwag4ks_subroutine(cycfeat=False,use_gt_in_cyc=True);
# class neko_HDOS2C_routine_CFmk7g3_cyc3ksgnfr(neko_HDOS2C_routine_CFmk7g3):
#     def set_etc(this, args):
#         this.maxT = args["maxT"];
#         this.core=neko_HDOS2C_routine_CFmk7g3_cyc3ksgnfr_core();
#
#
# class neko_HDOS2C_routine_CFmk7g3_cyc3ksnfrd_core(neko_HDOS2C_routine_CFmk7g3_rec_cyc_core):
#     def arm_submodules(this):
#         this.inflater = neko_inflater();
#         this.water_mod=neko_cwag4ks_subroutine(cycfeat=False,det_proto_for_rec=True);
#
# class neko_HDOS2C_routine_CFmk7g3_cyc3ksnfrd(neko_HDOS2C_routine_CFmk7g3):
#     def set_etc(this, args):
#         this.maxT = args["maxT"];
#         this.core=neko_HDOS2C_routine_CFmk7g3_cyc3ksnfrd_core();
#
#
#
# class neko_HDOS2C_routine_CFmk7g3_rec_cyc3ksg_core(neko_HDOS2C_routine_CFmk7g3_rec_cyc_core):
#     def arm_submodules(this):
#         this.inflater = neko_inflater();
#         this.water_mod=neko_cwag4ks_subroutine(use_gt_in_cyc=True);
#
# class neko_HDOS2C_routine_CFmk7g3_rec_cyc3ksg(neko_HDOS2C_routine_CFmk7g3):
#     def set_etc(this, args):
#         this.maxT = args["maxT"];
#         this.core=neko_HDOS2C_routine_CFmk7g3_rec_cyc3ksg_core();
#
# class neko_HDOS2C_routine_CFmk7g3_rec_cyc3ksdg_core(neko_HDOS2C_routine_CFmk7g3_rec_cyc_core):
#     def arm_submodules(this):
#         this.inflater = neko_inflater();
#         this.water_mod=neko_cwag4ks_subroutine(use_gt_in_cyc=True,det_proto_for_rec=True);
#
# class neko_HDOS2C_routine_CFmk7g3_rec_cyc3ksdg(neko_HDOS2C_routine_CFmk7g3):
#     def set_etc(this, args):
#         this.maxT = args["maxT"];
#         this.core=neko_HDOS2C_routine_CFmk7g3_rec_cyc3ksdg_core();
#
#
# class neko_HDOS2C_routine_CFmk7g3_rec_cyc3ks128_core(neko_HDOS2C_routine_CFmk7g3_rec_cyc_core):
#     def arm_submodules(this):
#         this.inflater = neko_inflater();
#         this.water_mod=neko_cwag4ks_subroutine(cnt=128);
#
# class neko_HDOS2C_routine_CFmk7g3_rec_cyc3ks128(neko_HDOS2C_routine_CFmk7g3):
#     def set_etc(this, args):
#         this.maxT = args["maxT"];
#         this.core=neko_HDOS2C_routine_CFmk7g3_rec_cyc3ks128_core();
#
# class neko_HDOS2C_routine_CFmk7g3_rec_cyc3ksa_core(neko_HDOS2C_routine_CFmk7g3_rec_cyc_core):
#     def arm_submodules(this):
#         this.inflater = neko_inflater();
#         this.water_mod=neko_cwag4ks_subroutine(cycshuf=False);
#
# class neko_HDOS2C_routine_CFmk7g3_rec_cyc3ksa(neko_HDOS2C_routine_CFmk7g3):
#     def set_etc(this, args):
#         this.maxT = args["maxT"];
#         this.core=neko_HDOS2C_routine_CFmk7g3_rec_cyc3ksa_core();
# class neko_HDOS2C_routine_CFmk7g3_rec_cyc3ksad_core(neko_HDOS2C_routine_CFmk7g3_rec_cyc_core):
#     def arm_submodules(this):
#         this.inflater = neko_inflater();
#         this.water_mod=neko_cwag4ks_subroutine(cycshuf=False,det_proto_for_rec=True);
#
# class neko_HDOS2C_routine_CFmk7g3_rec_cyc3ksad(neko_HDOS2C_routine_CFmk7g3):
#     def set_etc(this, args):
#         this.maxT = args["maxT"];
#         this.core=neko_HDOS2C_routine_CFmk7g3_rec_cyc3ksa_core();
#
# class neko_HDOS2C_routine_CFmk7g3_rec_cyc3ksak_core(neko_HDOS2C_routine_CFmk7g3_rec_cyc_core):
#     def arm_submodules(this):
#         this.inflater = neko_inflater();
#         this.water_mod=neko_cwag4ks_subroutine(cycshuf=False,cycknown=True);
#
# class neko_HDOS2C_routine_CFmk7g3_rec_cyc3ksak(neko_HDOS2C_routine_CFmk7g3):
#     def set_etc(this, args):
#         this.maxT = args["maxT"];
#         this.core=neko_HDOS2C_routine_CFmk7g3_rec_cyc3ksak_core();
#
# class neko_HDOS2C_routine_CFmk7g3_rec_cyc3ksakd_core(neko_HDOS2C_routine_CFmk7g3_rec_cyc_core):
#     def arm_submodules(this):
#         this.inflater = neko_inflater();
#         this.water_mod=neko_cwag4ks_subroutine(cycshuf=False,cycknown=True,det_proto_for_rec=True);
#
# class neko_HDOS2C_routine_CFmk7g3_rec_cyc3ksakd(neko_HDOS2C_routine_CFmk7g3):
#     def set_etc(this, args):
#         this.maxT = args["maxT"];
#         this.core=neko_HDOS2C_routine_CFmk7g3_rec_cyc3ksak_core();
#
# class neko_HDOS2C_routine_CFmk7g3_rec_cyc3kss_core(neko_HDOS2C_routine_CFmk7g3_rec_cyc_core):
#     def arm_submodules(this):
#         this.inflater = neko_inflater();
#         this.water_mod=neko_cwag4ks_subroutine(use_gt_in_cyc=True);
#
# class neko_HDOS2C_routine_CFmk7g3_rec_cyc3kss(neko_HDOS2C_routine_CFmk7g3):
#     def set_etc(this, args):
#         this.maxT = args["maxT"];
#         this.core=neko_HDOS2C_routine_CFmk7g3_rec_cyc3kss_core();
#
# class neko_HDOS2C_routine_CFmk7g3_rec_cyc3ksas_core(neko_HDOS2C_routine_CFmk7g3_rec_cyc_core):
#     def arm_submodules(this):
#         this.inflater = neko_inflater();
#         this.water_mod=neko_cwag4ks_subroutine(cycshuf=False,use_gt_in_cyc=True);
#
#
# class neko_HDOS2C_routine_CFmk7g3_rec_cyc3ksasd_core(neko_HDOS2C_routine_CFmk7g3_rec_cyc_core):
#     def arm_submodules(this):
#         this.inflater = neko_inflater();
#         this.water_mod=neko_cwag4ks_subroutine(cycshuf=False,det_proto_for_rec=True);
#
# class neko_HDOS2C_routine_CFmk7g3_rec_cyc3ksasd(neko_HDOS2C_routine_CFmk7g3):
#     def set_etc(this, args):
#         this.maxT = args["maxT"];
#         this.core=neko_HDOS2C_routine_CFmk7g3_rec_cyc3ksasd_core();
#
#
# class neko_HDOS2C_routine_CFmk7g3_rec_cyc3ksas(neko_HDOS2C_routine_CFmk7g3):
#     def set_etc(this, args):
#         this.maxT = args["maxT"];
#         this.core=neko_HDOS2C_routine_CFmk7g3_rec_cyc3ksas_core();
#
#
# class neko_HDOS2C_routine_CFmk7g3R_rec_cyc3ks_core(neko_HDOS2C_routine_CFmk7g3_rec_cyc_core):
#     def arm_submodules(this):
#         this.inflater = neko_inflater();
#         this.water_mod=neko_cwag4ks_subroutine(featnorml=0.1);
#
# class neko_HDOS2C_routine_CFmk7g3R_rec_cyc3ks(neko_HDOS2C_routine_CFmk7g3):
#     def set_etc(this, args):
#         this.maxT = args["maxT"];
#         this.core=neko_HDOS2C_routine_CFmk7g3R_rec_cyc3ks_core();
#
#
# class neko_HDOS2C_routine_CFmk7g3_rec_cyc3ksa128_core(neko_HDOS2C_routine_CFmk7g3_rec_cyc_core):
#     def arm_submodules(this):
#         this.inflater = neko_inflater();
#         this.water_mod=neko_cwag4ks_subroutine(cycshuf=False,cnt=128);
#
# class neko_HDOS2C_routine_CFmk7g3_rec_cyc3ksa128(neko_HDOS2C_routine_CFmk7g3):
#     def set_etc(this, args):
#         this.maxT = args["maxT"];
#         this.core=neko_HDOS2C_routine_CFmk7g3_rec_cyc3ksa128_core();
#
#
#
#
#
# class neko_cwag4ksb_subroutine:
#     def __init__(this,cnt=64,wprecon=1,withfrecon=True,detach_proto_mva=False,domx_shuf=False,domx_proto=True,domx_feat=True,
#                  featnorml=0.,cycshuf=True,use_gt_in_cyc=False,det_proto_for_rec=False,has_recon_cyc=True,has_proto_cyc=True,feat_only_magic=False,detach_frec_classifier=True):
#         this.CNT=cnt;
#         this.wprecon=wprecon;
#         this.withfrecon=withfrecon;
#         this.mva_detach_proto=detach_proto_mva;
#         this.domx_shuf=domx_shuf;
#         this.domx_proto=domx_proto;
#         this.domx_feat=domx_feat;
#         this.featnorml=featnorml;
#         this.cycshuf=cycshuf;
#         this.use_gt_in_cyc=use_gt_in_cyc;
#         this.det_proto_for_rec=det_proto_for_rec;
#         this.has_recon_cyc=has_recon_cyc;
#         this.has_proto_cyc=has_proto_cyc;
#         this.feat_only_magic=feat_only_magic;
#         this.detach_frec_classifier=detach_frec_classifier;
#
#
#
#     def get_chunk(this,rec_chunk,mylength):
#         if(len(rec_chunk)==0):
#             return [0,mylength];
#         else:
#             return [rec_chunk[-1][-1],rec_chunk[-1][-1]+mylength];
#
#
#     def fp_impl(this,fout_emb,proto_,normprotos_,plabels,label_flatten,device,modular_dict,tdict=None):
#         if(this.featnorml<0.000009):
#             loss=torch.tensor(0,device=device).float();
#             terms = {};
#         else:
#             loss=trnf.relu(fout_emb.norm(p=2,dim=1)-3,inplace=False).mean()*this.featnorml;
#             terms = {"fnorm":loss.item()};
#
#
#         cyc_ims = [];
#         cyc_labels = [];
#         psel=torch.randperm(proto_.shape[0])[:this.CNT].to(proto_.device);
#         proto=proto_[psel];
#         cls_proto=[proto_];
#         cls_plabel=[plabels[:-1].to(proto_.device)];
#         normprotos_cu=torch.cat(normprotos_).to(proto_.device);
#         normprotos=normprotos_cu[psel];
#         normprotos_label = plabels[psel].to(proto_.device);
#
#
#         if(modular_dict["p_recon"]!="NEP_skipped_NEP"):
#             rec_list = [];
#             rec_label=[];
#             rec_dict={};
#             rec_trunks=[];
#             cyc_ims.append(normprotos);
#             cyc_labels.append(normprotos_label);
#
#             if(modular_dict["p_recon_loss"]!="NEP_skipped_NEP"):
#                 rec_list.append(proto)
#                 rec_trunks.append(this.get_chunk(rec_trunks,proto.shape[0]));
#                 rec_label.append(normprotos_label);
#                 rec_dict["proto"]=len(rec_dict);
#
#             if (modular_dict["recon_char_pred"] != "NEP_skipped_NEP"):
#                 shufed_ids = torch.randperm(fout_emb.shape[0]);
#                 flab_shufed = label_flatten[shufed_ids];
#                 fsel_id = shufed_ids[flab_shufed != plabels[-1]];
#                 fsel_id = fsel_id[:this.CNT].to(proto_.device)
#                 normed_fout = trnf.normalize(fout_emb[fsel_id], dim=-1, p=2);
#                 flab = label_flatten[fsel_id].to(proto_.device);
#
#                 if (modular_dict["f_recon"] == "NEP_skipped_NEP"):
#                     rec_label.append(flab);
#                     rec_dict["feat"] = len(rec_list);
#                     rec_trunks.append(this.get_chunk(rec_trunks, normed_fout.shape[0]));
#                     rec_list.append(normed_fout);
#
#
#             if (modular_dict["shuf_img"] != "NEP_skipped_NEP"):
#                 shufp, mapping = modular_dict["shuf_img"](normprotos);
#                 pshufp = modular_dict["shuf_proto"]([shufp], use_sp=False);
#                 if(this.cycshuf):
#                     labs = torch.range(0, len(shufp) - 1, device=shufp.device, dtype=plabels.dtype) + plabels[-1];
#                     cls_plabel.append(labs)
#                     cls_proto.append(pshufp);
#                     rec_label.append(labs);
#                     cyc_ims.append(shufp);
#                     cyc_labels.append(labs);
#                 else:
#                     # simply ignore them.
#                     pass;
#                     # labs=torch.zeros(len(shufp), device=shufp.device, dtype=plabels.dtype)-1;
#                     # rec_label.append(torch.zeros_like(labs)-1)
#                 rec_dict["shuf"] = len(rec_list);
#                 rec_trunks.append(this.get_chunk(rec_trunks,pshufp.shape[0]));
#                 rec_list.append(pshufp);
#
#             recons = modular_dict["p_recon"](torch.cat(rec_list).unsqueeze(-1).unsqueeze(-1));
#             if (this.wprecon > 0 and modular_dict["p_recon_loss"]!="NEP_skipped_NEP"):
#                 l_precon = modular_dict["p_recon_loss"](recons[rec_trunks[rec_dict["proto"]][0]:rec_trunks[rec_dict["proto"]][1]], normprotos);
#                 loss += l_precon;
#                 terms["p_recon"] = l_precon.item();
#
#             if (modular_dict["shuf_img"] != "NEP_skipped_NEP"):
#                 l_precon_shuf = modular_dict["shuf_recon_loss"](recons[rec_trunks[rec_dict["shuf"]][0]:rec_trunks[rec_dict["shuf"]][1]], shufp);
#                 loss += l_precon_shuf;
#                 terms["shuf_recon"] = l_precon_shuf.item();
#                 # if (modular_dict["shuf_part_recon_loss"] != "NEP_skipped_NEP"):
#                 #     l_shufrec = modular_dict["shuf_part_recon_loss"](shufp, pshufp);
#                 #     loss += l_shufrec;
#                 #     terms["shuf_part_rec"] = l_shufrec.item();
#                 # A soft link to p_recon can be used here.
#
#             if (this.withfrecon and modular_dict["recon_char_fe"] != "NEP_skipped_NEP"):
#
#                 cls_plabel_c=torch.cat(cls_plabel);
#                 # hard wired
#                 # avoid models "remembering"  weird patterns
#                 p = torch.cat(cls_proto);
#
#                 if(this.has_proto_cyc):
#                     cyc_labels = torch.cat(cyc_labels);
#                     locyc = modular_dict["recon_char_fe"](torch.cat(cyc_ims));
#                     f_recon_logitc = modular_dict["recon_char_pred"](locyc, p, cls_plabel_c);
#                     l_freconc, _ = modular_dict["f_recon_loss"](None, f_recon_logitc,cyc_labels);
#                     loss += l_freconc;
#                     terms["cyc_tr"] = l_freconc.item();
#
#                 # protos do not learn from reconstructed, which might include weird embeddings
#                 if(this.has_recon_cyc):
#                     rec_labels = torch.cat(rec_label);
#                     recons = recons[:rec_labels.shape[0]];
#                     if (this.feat_only_magic):
#                         rec_labels = rec_labels[64:];
#                         recons = recons[64:];
#                     if(this.detach_frec_classifier):
#                         lorec = modular_dict["recon_char_fe"].dnattach_fwd(recons);
#                     else:
#                         lorec = modular_dict["recon_char_fe"](recons);
#                     if(this.det_proto_for_rec):
#                         f_recon_logitr = modular_dict["recon_char_pred"](lorec, p.detach(), cls_plabel_c);
#                     else:
#                         f_recon_logitr = modular_dict["recon_char_pred"](lorec, p, cls_plabel_c);
#                     l_freconr, _ = modular_dict["f_recon_loss"](None, f_recon_logitr,rec_labels);
#                     terms["cyc"] = l_freconr.item();
#
#             # if(modular_dict["fpm_recon_loss"]!="NEP_skipped_NEP"):
#             #     f_recon = recons[rec_trunks[rec_dict["feat"]][0]:rec_trunks[rec_dict["feat"]][1]];
#             #     l_fpm = modular_dict["fpm_recon_loss"](f_recon, normprotos_cu, plabels[:-1], flab, dbgkey=None).mean();
#             #     terms["fpm_recon"]=l_fpm.item();
#             #     loss+=l_fpm;
#             else:
#                 pass;
#
#
#         if(modular_dict["dom_mix"]!="NEP_skipped_NEP"):
#             dta=[]; # domain targets
#             if(this.domx_feat==True):
#                 dta.append(normed_fout);
#             elif(this.domx_feat=="detached"):
#                 dta.append(normed_fout.detach());
#
#             if(this.domx_proto==True):
#                 dta.append(proto);
#             elif (this.domx_proto == "detached"):
#                 dta.append(proto.detach());
#
#             if(this.domx_shuf==True):
#                 dta.append(pshufp);
#
#             l_dommix=modular_dict["dom_mix"](dta);
#             loss += l_dommix;
#             terms["dom_mix"] = l_dommix.item();
#         return loss,terms;
#
#
# class neko_HDOS2C_routine_CFmk7g3_rec_cyc3ksasbdf_core(neko_HDOS2C_routine_CFmk7g3_rec_cyc_core):
#     def arm_submodules(this):
#         this.inflater = neko_inflater();
#         this.water_mod=neko_cwag4ksb_subroutine(cycshuf=False,use_gt_in_cyc=False,det_proto_for_rec=False,has_proto_cyc=False);
#
# class neko_HDOS2C_routine_CFmk7g3_rec_cyc3ksasbd(neko_HDOS2C_routine_CFmk7g3):
#     def set_etc(this, args):
#         this.maxT = args["maxT"];
#         this.core=neko_HDOS2C_routine_CFmk7g3_rec_cyc3ksasbd_core();
#
# class neko_HDOS2C_routine_CFmk7g3_rec_cyc3ksasbd_core(neko_HDOS2C_routine_CFmk7g3_rec_cyc_core):
#     def arm_submodules(this):
#         this.inflater = neko_inflater();
#         this.water_mod=neko_cwag4ksb_subroutine(cycshuf=False,use_gt_in_cyc=True,det_proto_for_rec=True);
#
# class neko_HDOS2C_routine_CFmk7g3_rec_cyc3ksasbd(neko_HDOS2C_routine_CFmk7g3):
#     def set_etc(this, args):
#         this.maxT = args["maxT"];
#         this.core=neko_HDOS2C_routine_CFmk7g3_rec_cyc3ksasbd_core();
#
#
# class neko_HDOS2C_routine_CFmk7g3_rec_cyc3ksasbs_core(neko_HDOS2C_routine_CFmk7g3_rec_cyc_core):
#     def arm_submodules(this):
#         this.inflater = neko_inflater();
#         this.water_mod=neko_cwag4ksb_subroutine(cycshuf=True,use_gt_in_cyc=True);
#
# class neko_HDOS2C_routine_CFmk7g3_rec_cyc3ksasbs(neko_HDOS2C_routine_CFmk7g3):
#     def set_etc(this, args):
#         this.maxT = args["maxT"];
#         this.core=neko_HDOS2C_routine_CFmk7g3_rec_cyc3ksasbds_core();
#
# class neko_HDOS2C_routine_CFmk7g3_rec_cyc3ksasbds_core(neko_HDOS2C_routine_CFmk7g3_rec_cyc_core):
#     def arm_submodules(this):
#         this.inflater = neko_inflater();
#         this.water_mod=neko_cwag4ksb_subroutine(cycshuf=True,use_gt_in_cyc=True,det_proto_for_rec=True);
#
# class neko_HDOS2C_routine_CFmk7g3_rec_cyc3ksasbds(neko_HDOS2C_routine_CFmk7g3):
#     def set_etc(this, args):
#         this.maxT = args["maxT"];
#         this.core=neko_HDOS2C_routine_CFmk7g3_rec_cyc3ksasbds_core();
#
# class neko_HDOS2C_routine_CFmk7g3_rec_cyc3ksasbd_core_cycrp(neko_HDOS2C_routine_CFmk7g3_rec_cyc_core):
#     def arm_submodules(this):
#         this.inflater = neko_inflater();
#         this.water_mod=neko_cwag4ksb_subroutine(cycshuf=False,use_gt_in_cyc=True,det_proto_for_rec=True,has_proto_cyc=True,has_recon_cyc=False);
#
# class neko_HDOS2C_routine_CFmk7g3_rec_cyc3ksasbd_cycrp(neko_HDOS2C_routine_CFmk7g3):
#     def set_etc(this, args):
#         this.maxT = args["maxT"];
#         this.core=neko_HDOS2C_routine_CFmk7g3_rec_cyc3ksasbd_core_cycrp();
# class neko_HDOS2C_routine_CFmk7g3_rec_cyc3ksasbd_core_cycrx(neko_HDOS2C_routine_CFmk7g3_rec_cyc_core):
#     def arm_submodules(this):
#         this.inflater = neko_inflater();
#         this.water_mod=neko_cwag4ksb_subroutine(cycshuf=False,use_gt_in_cyc=True,det_proto_for_rec=True,has_proto_cyc=False,has_recon_cyc=False);
#
# class neko_HDOS2C_routine_CFmk7g3_rec_cyc3ksasbd_cycrx(neko_HDOS2C_routine_CFmk7g3):
#     def set_etc(this, args):
#         this.maxT = args["maxT"];
#         this.core=neko_HDOS2C_routine_CFmk7g3_rec_cyc3ksasbd_core_cycrx();
# class neko_HDOS2C_routine_CFmk7g3_rec_cyc3ksasbd_core_cycrfp(neko_HDOS2C_routine_CFmk7g3_rec_cyc_core):
#     def arm_submodules(this):
#         this.inflater = neko_inflater();
#         this.water_mod=neko_cwag4ksb_subroutine(cycshuf=False,use_gt_in_cyc=True,det_proto_for_rec=True,has_proto_cyc=False);
#
# class neko_HDOS2C_routine_CFmk7g3_rec_cyc3ksasbd_cycrfp(neko_HDOS2C_routine_CFmk7g3):
#     def set_etc(this, args):
#         this.maxT = args["maxT"];
#         this.core=neko_HDOS2C_routine_CFmk7g3_rec_cyc3ksasbd_core_cycrfp();
# class neko_HDOS2C_routine_CFmk7g3_rec_cyc3ksasbd_core_cycrfpnd(neko_HDOS2C_routine_CFmk7g3_rec_cyc_core):
#     def arm_submodules(this):
#         this.inflater = neko_inflater();
#         this.water_mod=neko_cwag4ksb_subroutine(cycshuf=False,use_gt_in_cyc=True,det_proto_for_rec=False,has_proto_cyc=False);
#
# class neko_HDOS2C_routine_CFmk7g3_rec_cyc3ksasbd_cycrfpnd(neko_HDOS2C_routine_CFmk7g3):
#     def set_etc(this, args):
#         this.maxT = args["maxT"];
#         this.core=neko_HDOS2C_routine_CFmk7g3_rec_cyc3ksasbd_core_cycrfpnd();
# class neko_HDOS2C_routine_CFmk7g3_rec_cyc3ksasbd_core_cycrf(neko_HDOS2C_routine_CFmk7g3_rec_cyc_core):
#     def arm_submodules(this):
#         this.inflater = neko_inflater();
#         this.water_mod=neko_cwag4ksb_subroutine(cycshuf=False,use_gt_in_cyc=True,det_proto_for_rec=True,has_proto_cyc=False,feat_only_magic=True);
#
# class neko_HDOS2C_routine_CFmk7g3_rec_cyc3ksasbd_cycrf(neko_HDOS2C_routine_CFmk7g3):
#     def set_etc(this, args):
#         this.maxT = args["maxT"];
#         this.core=neko_HDOS2C_routine_CFmk7g3_rec_cyc3ksasbd_core_cycrf();
# class neko_HDOS2C_routine_CFmk7g3_rec_cyc3ksasbd_core_cycrftfr(neko_HDOS2C_routine_CFmk7g3_rec_cyc_core):
#     def arm_submodules(this):
#         this.inflater = neko_inflater();
#         this.water_mod=neko_cwag4ksb_subroutine(cycshuf=False,use_gt_in_cyc=True,det_proto_for_rec=True,has_proto_cyc=False,feat_only_magic=True,detach_frec_classifier=False);
#
# class neko_HDOS2C_routine_CFmk7g3_rec_cyc3ksasbd_cycrftfr(neko_HDOS2C_routine_CFmk7g3):
#     def set_etc(this, args):
#         this.maxT = args["maxT"];
#         this.core=neko_HDOS2C_routine_CFmk7g3_rec_cyc3ksasbd_core_cycrftfr();
#
# class neko_HDOS2C_routine_CFmk7g3_rec_cyc3ksasbd_core_cycrfnd(neko_HDOS2C_routine_CFmk7g3_rec_cyc_core):
#     def arm_submodules(this):
#         this.inflater = neko_inflater();
#         this.water_mod=neko_cwag4ksb_subroutine(cycshuf=False,use_gt_in_cyc=True,det_proto_for_rec=False,has_proto_cyc=False,feat_only_magic=True);
#
# class neko_HDOS2C_routine_CFmk7g3_rec_cyc3ksasbd_cycrfnd(neko_HDOS2C_routine_CFmk7g3):
#     def set_etc(this, args):
#         this.maxT = args["maxT"];
#         this.core=neko_HDOS2C_routine_CFmk7g3_rec_cyc3ksasbd_core_cycrfnd();
#
#
# class neko_HDOS2C_routine_CFmk7g3_rec_cyc3ksasbd_core_cycrfndtfr(neko_HDOS2C_routine_CFmk7g3_rec_cyc_core):
#     def arm_submodules(this):
#         this.inflater = neko_inflater();
#         this.water_mod=neko_cwag4ksb_subroutine(cycshuf=False,use_gt_in_cyc=True,det_proto_for_rec=False,has_proto_cyc=False,feat_only_magic=True,detach_frec_classifier=False);
#
# class neko_HDOS2C_routine_CFmk7g3_rec_cyc3ksasbd_cycrfndtfr(neko_HDOS2C_routine_CFmk7g3):
#     def set_etc(this, args):
#         this.maxT = args["maxT"];
#         this.core=neko_HDOS2C_routine_CFmk7g3_rec_cyc3ksasbd_core_cycrfndtfr();
#
#
#
# # Conditional cycloss
# class neko_cwag4kscc_subroutine:
#     def __init__(this,cnt=64,wprecon=1,withfrecon=True,detach_proto_mva=False,domx_shuf=False,domx_proto=True,domx_feat=True,featnorml=0.,cycshuf=True):
#         this.CNT=cnt;
#         this.wprecon=wprecon;
#         this.withfrecon=withfrecon;
#         this.mva_detach_proto=detach_proto_mva;
#         this.domx_shuf=domx_shuf;
#         this.domx_proto=domx_proto;
#         this.domx_feat=domx_feat;
#         this.featnorml=featnorml;
#         this.cycshuf=cycshuf;
#
#     def get_chunk(this,rec_chunk,mylength):
#         if(len(rec_chunk)==0):
#             return [0,mylength];
#         else:
#             return [rec_chunk[-1][-1],rec_chunk[-1][-1]+mylength];
#
#
#     def fp_impl(this,fout_emb,proto_,normprotos_,plabels,label_flatten,device,modular_dict,tdict=None):
#         if(this.featnorml<0.000009):
#             loss=torch.tensor(0,device=device).float();
#             terms = {};
#         else:
#             loss=trnf.relu(fout_emb.norm(p=2,dim=1)-3,inplace=False).mean()*this.featnorml;
#             terms = {"fnorm":loss.item()};
#
#         psel=torch.randperm(proto_.shape[0])[:this.CNT].to(proto_.device);
#         proto=proto_[psel];
#         cls_proto=[proto_];
#         cls_plabel=[plabels[:-1].to(proto_.device)];
#         normprotos_cu=torch.cat(normprotos_).to(proto_.device);
#         normprotos=normprotos_cu[psel];
#
#         if(modular_dict["p_recon"]!="NEP_skipped_NEP"):
#             rec_list = [];
#             rec_label=[];
#             rec_dict={};
#             rec_trunks=[];
#
#             if(modular_dict["p_recon_loss"]!="NEP_skipped_NEP"):
#                 rec_list.append(proto)
#                 rec_trunks.append(this.get_chunk(rec_trunks,proto.shape[0]));
#                 rec_label.append(plabels[psel].to(proto_.device))
#                 rec_dict["proto"]=len(rec_dict);
#
#             if((modular_dict["recon_char_fe"]!="NEP_skipped_NEP")or (modular_dict["fpm_recon_loss"]!="NEP_skipped_NEP")):
#                 fsela = torch.randperm(fout_emb.shape[0]);
#                 flaba = label_flatten[fsela];
#                 fsel = fsela[flaba != plabels[-1]];
#                 fsel = fsel[:this.CNT].to(proto_.device)
#                 normed_fout = trnf.normalize(fout_emb[fsel], dim=-1, p=2);
#
#                 flab = label_flatten[fsel];
#                 rec_label.append(plabels[flab].to(proto_.device));
#                 rec_dict["feat"]=len(rec_list);
#                 rec_trunks.append(this.get_chunk(rec_trunks,normed_fout.shape[0]));
#                 rec_list.append(normed_fout);
#
#             if (modular_dict["shuf_img"] != "NEP_skipped_NEP"):
#                 shufp, mapping = modular_dict["shuf_img"](normprotos);
#                 pshufp = modular_dict["shuf_proto"]([shufp], use_sp=False);
#                 if(this.cycshuf):
#                     labs = torch.range(0, len(shufp) - 1, device=shufp.device, dtype=plabels.dtype) + plabels[-1];
#                     cls_plabel.append(labs)
#                     cls_proto.append(pshufp);
#                     rec_label.append(labs);
#                 else:
#                     # simply ignore them.
#                     labs=torch.zeros(len(shufp), device=shufp.device, dtype=plabels.dtype)-1;
#                     rec_label.append(torch.zeros_like(labs))
#                 rec_dict["shuf"] = len(rec_list);
#                 rec_trunks.append(this.get_chunk(rec_trunks,pshufp.shape[0]));
#                 rec_list.append(pshufp);
#
#
#             recons = modular_dict["p_recon"](torch.cat(rec_list).unsqueeze(-1).unsqueeze(-1));
#             if (this.wprecon > 0 and modular_dict["p_recon_loss"]!="NEP_skipped_NEP"):
#                 l_precon = modular_dict["p_recon_loss"](recons[rec_trunks[rec_dict["proto"]][0]:rec_trunks[rec_dict["proto"]][1]], normprotos);
#                 loss += l_precon;
#                 terms["p_recon"] = l_precon.item();
#
#             if (modular_dict["shuf_img"] != "NEP_skipped_NEP"):
#                 l_precon_shuf = modular_dict["shuf_recon_loss"](recons[rec_trunks[rec_dict["shuf"]][0]:rec_trunks[rec_dict["shuf"]][1]], shufp);
#                 loss += l_precon_shuf;
#                 terms["shuf_recon"] = l_precon_shuf.item();
#                 if (modular_dict["shuf_part_recon_loss"] != "NEP_skipped_NEP"):
#                     l_shufrec = modular_dict["shuf_part_recon_loss"](shufp, pshufp);
#                     loss += l_shufrec;
#                     terms["shuf_part_rec"] = l_shufrec.item();
#                 # A soft link to p_recon can be used here.
#
#             if (this.withfrecon and modular_dict["recon_char_fe"] != "NEP_skipped_NEP"):
#                 labels=torch.cat(rec_label);
#                 cls_plabel_c=torch.cat(cls_plabel)
#                 # hard wired
#                 pred_f_recon = modular_dict["recon_char_fe"](recons);
#                 f_recon_logit = modular_dict["recon_char_pred"](pred_f_recon, torch.cat(cls_proto), cls_plabel_c);
#                 l_frecon, _ = modular_dict["f_recon_loss"](None, f_recon_logit, labels);
#                 loss += l_frecon;
#                 terms["cyc"] = l_frecon.item();
#             # if(modular_dict["fpm_recon_loss"]!="NEP_skipped_NEP"):
#             #     f_recon = recons[rec_trunks[rec_dict["feat"]][0]:rec_trunks[rec_dict["feat"]][1]];
#             #     l_fpm = modular_dict["fpm_recon_loss"](f_recon, normprotos_cu, plabels[:-1], flab, dbgkey=None).mean();
#             #     terms["fpm_recon"]=l_fpm.item();
#             #     loss+=l_fpm;
#             else:
#                 pass;
#
#
#         if(modular_dict["dom_mix"]!="NEP_skipped_NEP"):
#             dta=[]; # domain targets
#             if(this.domx_feat==True):
#                 dta.append(normed_fout);
#             elif(this.domx_feat=="detached"):
#                 dta.append(normed_fout.detach());
#
#             if(this.domx_proto==True):
#                 dta.append(proto);
#             elif (this.domx_proto == "detached"):
#                 dta.append(proto.detach());
#
#             if(this.domx_shuf==True):
#                 dta.append(pshufp);
#
#             l_dommix=modular_dict["dom_mix"](dta);
#             loss += l_dommix;
#             terms["dom_mix"] = l_dommix.item();
#         return loss,terms;