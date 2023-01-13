from neko_sdk.MJT.default_config import get_default_model
from neko_2020nocr.dan.configs.pipelines_pami import get_bl_fe_args
from neko_sdk.chunked_resnet.res45 import neko_r45_binorm,neko_r45_binorm_orig,neko_r45_binorm_ptpt,neko_r45_binorm_tpt



def get_dan_r45_binorm(arg_dict,path,optim_path=None):
    args=get_bl_fe_args(arg_dict["ouch"],arg_dict["ich"]);
    args["bogo_names"]=arg_dict["bogo_names"];
    args["bn_names"]=arg_dict["bn_names"];

    return get_default_model(neko_r45_binorm,args,path,arg_dict["with_optim"],optim_path);


def config_fe_r45_binorm(ich,feat_ch,input_shape=None,cnt=2):
    bogo_names=[];
    bn_names=[];
    for i in range(cnt):
        bogo_names.append("res"+str(i+1));
        bn_names.append("bn"+str(i+1));

    return \
    {
        "modular": get_dan_r45_binorm,
        "provides_bogo_modules": bogo_names,
        "save_each": 20000,
        "args":
            {
                "bogo_names":bogo_names,
                "bn_names": bn_names,
                "ich": ich,
                "ouch": feat_ch,
                "with_optim": True,
                "input_shape": input_shape,
                "strides":None,
            }
    }

def get_dan_r45_binormXN(arg_dict,path,optim_path=None):
    args=get_bl_fe_args(arg_dict["ouch"],arg_dict["ich"]);
    args["bogo_names"]=arg_dict["bogo_names"];
    args["norm_names"]=arg_dict["norm_names"];
    args["norm_types"] = arg_dict["norm_types"];

    return get_default_model(neko_r45_binorm_origXN,args,path,arg_dict["with_optim"],optim_path);


def get_dan_r45_binorm_orig(arg_dict,path,optim_path=None):
    args=get_bl_fe_args(arg_dict["ouch"],arg_dict["ich"]);
    args["bogo_names"] = arg_dict["bogo_names"];
    args["bn_names"] = arg_dict["bn_names"];
    args["inplace"]=arg_dict["inplace"];
    args["bn_affine"]=arg_dict["bn_affine"];
    args["expf"]=arg_dict["expf"];
    args["drop"]=arg_dict["drop"];
    return get_default_model(neko_r45_binorm_orig,args,path,arg_dict["with_optim"],optim_path);

def get_dan_r45_binorm_tpt(arg_dict,path,optim_path=None):
    args=get_bl_fe_args(arg_dict["ouch"],arg_dict["ich"]);
    args["bogo_names"] = arg_dict["bogo_names"];
    args["bn_names"] = arg_dict["bn_names"];
    args["expf"]=arg_dict["expf"];
    return get_default_model(neko_r45_binorm_tpt,args,path,arg_dict["with_optim"],optim_path);
def get_dan_r45_binorm_ptpt(arg_dict,path,optim_path=None):
    args=get_bl_fe_args(arg_dict["ouch"],arg_dict["ich"]);
    args["bogo_names"] = arg_dict["bogo_names"];
    args["bn_names"] = arg_dict["bn_names"];
    args["expf"]=arg_dict["expf"];
    args["inplace"]=arg_dict["inplace"];
    args["drop"]=arg_dict["drop"];
    return get_default_model(neko_r45_binorm_ptpt,args,path,arg_dict["with_optim"],optim_path);
def get_dan_r45_binorm_ptptXN(arg_dict,path,optim_path=None):
    args=get_bl_fe_args(arg_dict["ouch"],arg_dict["ich"]);
    args["bogo_names"]=arg_dict["bogo_names"];
    args["norm_names"]=arg_dict["norm_names"];
    args["norm_types"] = arg_dict["norm_types"];
    args["expf"]=arg_dict["expf"];
    args["inplace"]=arg_dict["inplace"];
    args["drop"]=arg_dict["drop"];
    return get_default_model(neko_r45_binormXN_ptpt,args,path,arg_dict["with_optim"],optim_path);

def config_fe_r45_binorm_tpt(ich,feat_ch,input_shape=None,cnt=2,expf=1):
    bogo_names=[];
    bn_names=[];
    for i in range(cnt):
        bogo_names.append("res"+str(i+1));
        bn_names.append("bn"+str(i+1));

    return \
    {
        "modular": get_dan_r45_binorm_tpt,
        "provides_bogo_modules": bogo_names,
        "save_each": 20000,
        "args":
            {
                "bogo_names": bogo_names,
                "bn_names": bn_names,
                "ich": ich,
                "ouch": feat_ch,
                "expf":expf,
                "with_optim": True,
                "input_shape": input_shape,
                "strides":None,
            }
    }

def config_fe_r45_binorm_ptpt(ich, feat_ch, input_shape=None, cnt=2, expf=1, inplace=True,drop=0):
    bogo_names=[];
    bn_names=[];
    for i in range(cnt):
        bogo_names.append("res"+str(i+1));
        bn_names.append("bn"+str(i+1));

    return \
    {
        "modular": get_dan_r45_binorm_ptpt,
        "provides_bogo_modules": bogo_names,
        "save_each": 20000,
        "args":
            {
                "bogo_names": bogo_names,
                "bn_names": bn_names,
                "ich": ich,
                "ouch": feat_ch,
                "expf":expf,
                "with_optim": True,
                "input_shape": input_shape,
                "strides":None,
                "inplace":inplace,
                "drop": drop
            }
    }

def config_fe_r45_binorm_orig(ich,feat_ch,input_shape=None,cnt=2,expf=1,inplace=True,bn_affine=True,drop=0):
    bogo_names=[];
    bn_names=[];
    for i in range(cnt):
        bogo_names.append("res"+str(i+1));
        bn_names.append("bn"+str(i+1));

    return \
    {
        "modular": get_dan_r45_binorm_orig,
        "provides_bogo_modules": bogo_names,
        "save_each": 20000,
        "args":
            {
                "bogo_names": bogo_names,
                "bn_names": bn_names,
                "ich": ich,
                "ouch": feat_ch,
                "expf":expf,
                "with_optim": True,
                "input_shape": input_shape,
                "strides":None,
                "inplace":inplace,
                "bn_affine":bn_affine,
                "drop": drop
            }
    }


def config_fe_r45_binormBIN_orig(ich,feat_ch,input_shape=None,cnt=[2,2],expf=1,inplace=True,bn_affine=True,drop=0):
    bogo_names=[];
    norm_names=[];
    norm_types=[];
    for i in range(cnt[0]):
        bogo_names.append("res"+str(i+1));
        norm_names.append("bn"+str(i+1));
        norm_types.append("bn")
    for i in range(cnt[0],cnt[0]+cnt[1]):
        bogo_names.append("res"+str(i+1));
        norm_names.append("bn"+str(i+1));
        norm_types.append("in")
    return \
    {
        "modular": get_dan_r45_binormXN,
        "provides_bogo_modules": bogo_names,
        "save_each": 20000,
        "args":
            {
                "bogo_names": bogo_names,
                "norm_names": norm_names,
                "norm_types": norm_types,
                "ich": ich,
                "ouch": feat_ch,
                "expf":expf,
                "with_optim": True,
                "input_shape": input_shape,
                "strides":None,
                "inplace":inplace,
                "bn_affine":bn_affine,
                "drop": drop
            }
    }


def config_fe_r45_binormBIN_ptpt(ich, feat_ch, input_shape=None, cnt=[2,2], expf=1, inplace=True,drop=0):
    bogo_names=[];
    norm_names=[];
    norm_types=[];
    for i in range(cnt[0]):
        bogo_names.append("res" + str(i + 1));
        norm_names.append("bn" + str(i + 1));
        norm_types.append("bn")
    for i in range(cnt[0], cnt[0] + cnt[1]):
        bogo_names.append("res" + str(i + 1));
        norm_names.append("bn" + str(i + 1));
        norm_types.append("in")

    return \
    {
        "modular": get_dan_r45_binorm_ptptXN,
        "provides_bogo_modules": bogo_names,
        "save_each": 20000,
        "args":
            {
                "bogo_names": bogo_names,
                "norm_names": norm_names,
                "norm_types": norm_types,
                "ich": ich,
                "ouch": feat_ch,
                "expf":expf,
                "with_optim": True,
                "input_shape": input_shape,
                "strides":None,
                "inplace":inplace,
                "drop": drop
            }
    }

def get_dan_r45_binorm_forkhead_orig(arg_dict,path,optim_path=None):
    args=get_bl_fe_args(arg_dict["ouch"],arg_dict["ich"]);
    args["bogo_names"] = arg_dict["bogo_names"];
    args["bn_names"] = arg_dict["bn_names"];

    return get_default_model(neko_r45_binorm_orig_forkhead,args,path,arg_dict["with_optim"],optim_path);


def config_fe_r45_binorm_forkhead_orig(ich,feat_ch,input_shape=None,cnt=2):
    bogo_names=[];
    bn_names=[];
    for i in range(cnt):
        bogo_names.append("res"+str(i+1));
        bn_names.append("bn"+str(i+1));

    return \
    {
        "modular": get_dan_r45_binorm_forkhead_orig,
        "provides_bogo_modules": bogo_names,
        "save_each": 20000,
        "args":
            {
                "bogo_names": bogo_names,
                "bn_names": bn_names,
                "ich": ich,
                "ouch": feat_ch,
                "with_optim": True,
                "input_shape": input_shape,
                "strides":None,
            }
    }
def get_dan_r45_binorm_heavy_head(arg_dict,path,optim_path=None):
    args=get_bl_fe_args(arg_dict["ouch"],arg_dict["ich"]);
    args["bogo_names"] = arg_dict["bogo_names"];
    args["bn_names"] = arg_dict["bn_names"];

    return get_default_model(neko_r45_binorm_heavy_head,args,path,arg_dict["with_optim"],optim_path);

def config_fe_r45_binorm_heavy_head(ich,feat_ch,input_shape=None,cnt=2):
    bogo_names = [];
    bn_names = [];
    for i in range(cnt):
        bogo_names.append("res" + str(i + 1));
        bn_names.append("bn" + str(i + 1));
    return \
    {
        "modular": get_dan_r45_binorm_heavy_head,
        "provides_bogo_modules": bogo_names,
        "save_each": 20000,
        "args":
            {
                "bogo_names": bogo_names,
                "bn_names": bn_names,
                "ich": ich,
                "ouch": feat_ch,
                "with_optim": True,
                "input_shape": input_shape,
                "strides":None,
            }
    }

