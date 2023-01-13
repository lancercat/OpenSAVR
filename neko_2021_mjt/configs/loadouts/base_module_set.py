
from neko_sdk.MJT.eval_tasks.dan_eval_tasks import neko_odan_eval_tasks,neko_abstract_eval_tasks

def arm_base_eval_routine2(srcdst,tname,prefix,routine_type,log_path,maxT,measure_rej=False,override_dict=None):
    kvargs = {"prototyper_name": prefix + "prototyper",
              "sampler_name": prefix + "Latin_62_sampler",
              "feature_extractor_name": prefix + "feature_extractor_cco",
              "CAMname": prefix + "TA",
              "seq_name": prefix + "DTD",
              "pred_name": [prefix + "pred"],
              "loss_name": [prefix + "loss_cls_emb"],
              "image_name": "image",
              "label_name": "label",
              "log_path": log_path,
              "name": prefix + tname,
              "maxT": maxT,
              "measure_rej": measure_rej, }
    if(override_dict is not None):
        for k in override_dict:
            kvargs[k]=override_dict[k];
    return routine_type(
        **kvargs
    );
def arm_base_task_default2(srcdst,prefix,routine_type,maxT,te_meta_path,datasets,log_path,name="close_set_benchmarks",measure_rej=False,override_dict=None):
    te_routine={};
    te_routine=arm_base_eval_routine2(te_routine,"close_set_benchmark",
                                      prefix,routine_type,log_path,maxT,measure_rej=measure_rej,override_dict=override_dict)
    if(override_dict is not None):
        if("prototyper_name" in override_dict):
            pengine=override_dict["prototyper_name"];
        else:
            pengine=prefix + "prototyper"
    else:
        pengine = prefix + "prototyper"

    srcdst[prefix+name]={
        "type": neko_odan_eval_tasks,
        "protoname":pengine,
        "temeta":
            {
                "meta_path": te_meta_path,
                "case_sensitive": False,
            },
        "datasets":datasets,
        "routine_cfgs": te_routine,
    }
    return srcdst
#
# from neko_2021_mjt.configs.modules.config_fe_cco import config_fe_cco_thicc,\
#     config_fe_cco,config_fe_acog_thicc
# from neko_2021_mjt.configs.modules.config_fe_aof import config_fe_cco_thicc_std,config_fe_cco_thicc_sdlr
#
# from  neko_2021_mjt.configs.modules.config_fe_std import config_fe_r50,config_fe_r50f,config_fe_r45
#
# from neko_2021_mjt.configs.common_subs.arm_postfe import arm_rest_common,arm_rest_commonr34
# from neko_2021_mjt.configs.modules.config_cam import config_cam
# from neko_2021_mjt.configs.modules.config_cam import config_cam
# from neko_2021_mjt.configs.common_subs.arm_postfe import arm_rest_common
# from neko_2021_mjt.configs.modules.config_fe_cco import config_fe_cco_thicc
#
# # def arm_base_module_set(srcdst,prefix,maxT,capacity,feat_ch,tr_meta_path,expf=1):
# #     srcdst[prefix+"feature_extractor_cco"]= config_fe_cco(3, feat_ch,expf=expf);
# #     srcdst[prefix+"CAM"]= config_cam(maxT, feat_ch=feat_ch,expf=expf);
# #     srcdst = arm_rest_common(srcdst, prefix, maxT, capacity, feat_ch, tr_meta_path)
# #     return srcdst;
# def arm_baser50_module_set(srcdst,prefix,maxT,capacity,feat_ch,tr_meta_path):
#     input_shape = [64, 480];
#     srcdst[prefix + "feature_extractor_cco"] = config_fe_r50(3, [feat_ch//4,feat_ch//2,feat_ch]);
#     srcdst[prefix + "CAM"] = config_cam(maxT, feat_ch=feat_ch,scales=[
#                     [int(feat_ch//4), input_shape[0]//8, input_shape[1]//8],
#                     [int(feat_ch//2), input_shape[0]//16, input_shape[1]//16],
#                     [int(feat_ch), input_shape[0]//32, input_shape[1]//32],
#                 ],);
#     srcdst=arm_rest_common(srcdst,prefix,maxT,capacity,feat_ch,tr_meta_path)
#     return srcdst;
# def arm_baser50f_module_set(srcdst,prefix,maxT,capacity,feat_ch,tr_meta_path):
#     input_shape = [64, 480];
#     srcdst[prefix + "feature_extractor_cco"] = config_fe_r50f(3, [feat_ch//4,feat_ch//2,feat_ch]);
#     srcdst[prefix + "CAM"] = config_cam(maxT, feat_ch=feat_ch,scales=[
#                     [int(feat_ch//4), input_shape[0]//4, input_shape[1]//4],
#                     [int(feat_ch//2), input_shape[0]//8, input_shape[1]//8],
#                     [int(feat_ch), input_shape[0]//16, input_shape[1]//16],
#                 ],);
#     srcdst = arm_rest_common(srcdst, prefix, maxT, capacity, feat_ch, tr_meta_path)
#     return srcdst;
#
# def arm_base_module_set_dan_r45_r34(srcdst,prefix,maxT,capacity,feat_ch,tr_meta_path,expf=1):
#     srcdst[prefix+"feature_extractor_cco"]= config_fe_r45(3, feat_ch);
#     srcdst[prefix+"CAM"]= config_cam(maxT, feat_ch=feat_ch,scales=[
#                     [int(32), 16, 64],
#                     [int(128), 8, 32],
#                     [int(feat_ch), 8, 32]
#                 ]);
#     srcdst = arm_rest_commonr34(srcdst, prefix, maxT, capacity, feat_ch, tr_meta_path)
#     return srcdst;
#
# def arm_base_module_set_dan_r45(srcdst,prefix,maxT,capacity,feat_ch,tr_meta_path,expf=1,wemb=0.3):
#     srcdst[prefix+"feature_extractor_cco"]= config_fe_r45(3, feat_ch);
#     srcdst[prefix+"CAM"]= config_cam(maxT, feat_ch=feat_ch,scales=[
#                     [int(32), 16, 64],
#                     [int(128), 8, 32],
#                     [int(feat_ch), 8, 32]
#                 ]);
#     srcdst = arm_rest_common(srcdst, prefix, maxT, capacity, feat_ch, tr_meta_path,wemb=wemb)
#     return srcdst;
# def arm_base_module_set_dan_r45c(srcdst,prefix,maxT,capacity,feat_ch,tr_meta_path,expf=1):
#     srcdst[prefix+"feature_extractor_cco"]= config_fe_r45(3, feat_ch);
#     srcdst[prefix+"CAM"]= config_cam(maxT, feat_ch=feat_ch,scales=[
#                     [int(32), 16, 64],
#                     [int(128), 8, 32],
#                     [int(feat_ch), 8, 32]
#                 ]);
#     srcdst = arm_rest_common(srcdst, prefix, maxT, capacity, feat_ch, tr_meta_path)
#     return srcdst;
#
#
# def arm_base_module_set_thicc_acog(srcdst,prefix,maxT,capacity,feat_ch,tr_meta_path):
#     srcdst[prefix+"feature_extractor_cco"]= config_fe_acog_thicc(3, feat_ch);
#     srcdst[prefix+"CAM"]= config_cam(maxT, feat_ch=feat_ch, expf=1.5);
#     srcdst = arm_rest_common(srcdst, prefix, maxT, capacity, feat_ch, tr_meta_path)
#     return srcdst;
# def arm_base_module_set_thicc(srcdst,prefix,maxT,capacity,feat_ch,tr_meta_path):
#     srcdst[prefix+"feature_extractor_cco"]= config_fe_cco_thicc(3, feat_ch);
#     srcdst[prefix+"CAM"]= config_cam(maxT, feat_ch=feat_ch, expf=1.5);
#     srcdst = arm_rest_common(srcdst, prefix, maxT, capacity, feat_ch, tr_meta_path)
#     return srcdst;
#
# def arm_base_module_set_thicc_std(srcdst,prefix,maxT,capacity,feat_ch,tr_meta_path):
#     srcdst[prefix+"feature_extractor_cco"]= config_fe_cco_thicc_std(3, feat_ch);
#     # srcdst[prefix+"feature_extractor_cco"]= config_fe_cco_thicc(3, feat_ch);
#     srcdst[prefix+"CAM"]= config_cam(maxT, feat_ch=feat_ch, expf=1.5);
#     srcdst = arm_rest_common(srcdst, prefix, maxT, capacity, feat_ch, tr_meta_path)
#
#     return srcdst;
# def arm_base_module_set_thicc_sdlr(srcdst,prefix,maxT,capacity,feat_ch,tr_meta_path):
#     srcdst[prefix+"feature_extractor_cco"]= config_fe_cco_thicc_sdlr(3, feat_ch);
#     # srcdst[prefix+"feature_extractor_cco"]= config_fe_cco_thicc(3, feat_ch);
#     srcdst[prefix+"CAM"]= config_cam(maxT, feat_ch=feat_ch, expf=1.5);
#     srcdst = arm_rest_common(srcdst, prefix, maxT, capacity, feat_ch, tr_meta_path)
#
#     return srcdst;
# def arm_base_module_set_thicc_stdlr(srcdst,prefix,maxT,capacity,feat_ch,tr_meta_path):
#     srcdst[prefix+"feature_extractor_cco"]= config_fe_cco_thicc_std(3, feat_ch);
#     # srcdst[prefix+"feature_extractor_cco"]= config_fe_cco_thicc(3, feat_ch);
#     srcdst[prefix+"CAM"]= config_cam(maxT, feat_ch=feat_ch, expf=1.5);
#     srcdst = arm_rest_common(srcdst, prefix, maxT, capacity, feat_ch, tr_meta_path)
#
#     return srcdst;
#
#
#
#
#
# def arm_base_routine(srcdst, prefix, routine_type,maxT, log_path, log_each,dsprefix):
#     srcdst[prefix+"mjst"]= routine_type(
#         prototyper_name=prefix+"prototyper",
#         sampler_name=prefix+"Latin_62_sampler",
#         feature_extractor_name=prefix+"feature_extractor_cco",
#         CAMname=prefix+"CAM",
#         seq_name=prefix+"DTD",
#         pred_name=[prefix+"pred"],
#         loss_name=[prefix+"loss_cls_emb"],
#         image_name=dsprefix+"image",
#         label_name=dsprefix+"label",
#         log_path=log_path,
#         log_each=log_each,
#         name=prefix+"mjst",
#         maxT=maxT,
#     );
#     srcdst[prefix+"mjst"]["stream"]=prefix;
#     return srcdst;
#
#
#
# def arm_base_routine2(srcdst, prefix, routine_type,maxT, log_path, log_each,dsprefix,override_dict=None,rot_id="mjst"):
#
#     kvargs={
#         "prototyper_name":prefix+"prototyper",
#         "sampler_name":prefix+"Latin_62_sampler",
#         "feature_extractor_name":prefix+"feature_extractor_cco",
#         "CAMname":prefix+"TA",
#         "seq_name":prefix+"DTD",
#         "pred_name":[prefix+"pred"],
#         "loss_name":[prefix+"loss_cls_emb"],
#         "image_name":dsprefix+"image",
#         "label_name":dsprefix+"label",
#         "log_path":log_path,
#         "log_each":log_each,
#         "name":prefix+rot_id,
#         "maxT":maxT,}
#     if(override_dict is not None):
#         for k in override_dict:
#             kvargs[k]=override_dict[k];
#     srcdst[prefix+rot_id]= routine_type(
#         **kvargs
#     );
#
#     srcdst[prefix+rot_id]["stream"]=prefix;
#     return srcdst;
#
# def arm_base_routine2m(srcdst, prefix, routine_type,maxT, log_path, log_each,dsprefix,override_dict=None,rot_id="mjst"):
#     kvargs={
#         "prototyper_name":prefix+"prototyper",
#         "sampler_name":prefix+"Latin_62_sampler",
#         "feature_extractor_name":prefix+"feature_extractor_cco",
#         "CAMname":prefix+"TA",
#         "seq_name":prefix+"DTD",
#         "pred_name":[prefix+"pred"],
#         "loss_name":[prefix+"loss_cls_emb"],
#         "image_name":dsprefix+"image",
#         "label_name":dsprefix+"label",
#         "mask_name": dsprefix + "bmask",
#         "log_path":log_path,
#         "log_each":log_each,
#         "name":prefix+rot_id,
#         "maxT":maxT,}
#     if(override_dict is not None):
#         for k in override_dict:
#             kvargs[k]=override_dict[k];
#     srcdst[prefix+rot_id]= routine_type(
#         **kvargs
#     );
#
#     srcdst[prefix+rot_id]["stream"]=prefix;
#     return srcdst;
#
#
#
# def arm_base_eval_routine(srcdst,tname,prefix,routine_type,log_path,maxT):
#     return routine_type(
#         prototyper_name=prefix + "prototyper",
#         sampler_name=prefix + "Latin_62_sampler",
#         feature_extractor_name=prefix + "feature_extractor_cco",
#         CAMname=prefix + "CAM",
#         seq_name=prefix + "DTD",
#         pred_name=[prefix + "pred"],
#         loss_name=[prefix + "loss_cls_emb"],
#         image_name="image",
#         label_name="label",
#         log_path=log_path,
#         name=prefix + tname,
#         maxT=maxT,
#     );
# def arm_base_eval_routine2m(srcdst,tname,prefix,routine_type,log_path,maxT,measure_rej=False,override_dict=None):
#     kvargs = {"prototyper_name": prefix + "prototyper",
#               "sampler_name": prefix + "Latin_62_sampler",
#               "feature_extractor_name": prefix + "feature_extractor_cco",
#               "CAMname": prefix + "TA",
#               "seq_name": prefix + "DTD",
#               "pred_name": [prefix + "pred"],
#               "loss_name": [prefix + "loss_cls_emb"],
#               "image_name": "image",
#               "label_name": "label",
#               "mask_name": "bmask",
#               "log_path": log_path,
#               "name": prefix + tname,
#               "maxT": maxT,
#               "measure_rej": measure_rej, }
#     if(override_dict is not None):
#         for k in override_dict:
#             kvargs[k]=override_dict[k];
#     return routine_type(
#         **kvargs
#     )
# from neko_2021_mjt.eval_tasks.dan_eval_tasks import neko_odan_eval_tasks
#
# def arm_base_task_default(srcdst,prefix,routine_type,maxT,te_meta_path,datasets,log_path,name="close_set_benchmarks"):
#     te_routine={};
#     te_routine=arm_base_eval_routine(te_routine,"close_set_benchmark",prefix,routine_type,log_path,maxT)
#     srcdst[prefix+name]={
#         "type": neko_odan_eval_tasks,
#         "protoname": prefix+"prototyper",
#         "temeta":
#             {
#                 "meta_path": te_meta_path,
#                 "case_sensitive": False,
#             },
#         "datasets":datasets,
#         "routine_cfgs": te_routine,
#     }
#     return srcdst
# def arm_base_task_default2m(srcdst,prefix,routine_type,maxT,te_meta_path,datasets,log_path,name="close_set_benchmarks",measure_rej=False,override_dict=None):
#     te_routine={};
#     te_routine=arm_base_eval_routine2m(te_routine,"close_set_benchmark",prefix,
#                                        routine_type,log_path,maxT,measure_rej=measure_rej,override_dict=override_dict)
#     srcdst[prefix+name]={
#         "type": neko_odan_eval_tasks,
#         "protoname": prefix+"prototyper",
#         "temeta":
#             {
#                 "meta_path": te_meta_path,
#                 "case_sensitive": False,
#             },
#         "datasets":datasets,
#         "routine_cfgs": te_routine,
#     }
#     return srcdst
