from neko_sdk.encoders.chunked_resnet.neko_block_fe import make_init_layer_bn,make_init_layer_wo_bn,make_body_layer_bn,make_body_layer_wo_bn

from torch import nn


def res45_wo_bn(inpch,oupch,strides,frac=1,ochs=None,blkcnt=None,inplace=True,engine=nn.Conv2d):
    retlayers={};
    if(blkcnt is None):
        blkcnt = [None, 3, 4, 6, 6, 3];
    if(ochs is None):
        ochs = [int(32*frac),int(32 * frac), int(64 * frac), int(128 * frac), int(256 * frac), oupch]
    retlayers["0"]=make_init_layer_wo_bn(inpch[0],ochs[0],strides[0],inplace,engine=engine);
    retlayers["1"]=make_body_layer_wo_bn(ochs[0],blkcnt[1],ochs[1],1,strides[1],inplace,engine=engine);
    retlayers["2"] = make_body_layer_wo_bn(ochs[1], blkcnt[2], ochs[2], 1, strides[2],inplace,engine=engine);
    retlayers["3"] = make_body_layer_wo_bn(ochs[2], blkcnt[3], ochs[3], 1, strides[3],inplace,engine=engine);
    retlayers["4"] = make_body_layer_wo_bn(ochs[3], blkcnt[4], ochs[4], 1, strides[4],inplace,engine=engine);
    retlayers["5"] = make_body_layer_wo_bn(ochs[4], blkcnt[5], ochs[5], 1, strides[5],inplace,engine=engine);
    return retlayers;

def res45p_wo_bn(inpch,oupch,strides,frac=1,ochs=None,blkcnt=None,inplace=True):
    retlayers={};
    blkcnt = [None, 3, 4, 6, 6, 3];
    if(ochs is None):
        ochs = [int(32*frac),int(64 * frac), int(128 * frac), int(256 * frac), int(512 * frac), oupch]
    retlayers["0"]=make_init_layer_wo_bn(inpch[0],ochs[0],strides[0]);
    retlayers["1"]=make_body_layer_wo_bn(ochs[0],blkcnt[1],ochs[1],1,strides[1]);
    retlayers["2"] = make_body_layer_wo_bn(ochs[1], blkcnt[2], ochs[2], 1, strides[2]);
    retlayers["3"] = make_body_layer_wo_bn(ochs[2], blkcnt[3], ochs[3], 1, strides[3]);
    retlayers["4"] = make_body_layer_wo_bn(ochs[3], blkcnt[4], ochs[4], 1, strides[4]);
    retlayers["5"] = make_body_layer_wo_bn(ochs[4], blkcnt[5], ochs[5], 1, strides[5]);
    return retlayers;

def res45_bn(inpch,oupch,strides,frac=1,ochs=None,blkcnt=None,inplace=True,affine=True,engine=nn.BatchNorm2d):
    blkcnt = [None, 3, 4, 6, 6, 3];
    if ochs is None:
        ochs = [int(32*frac),int(32 * frac), int(64 * frac), int(128 * frac), int(256 * frac), oupch]
    retlayers = {};
    retlayers["0"] = make_init_layer_bn(ochs[0],affine=affine,engine=engine);
    retlayers["1"] = make_body_layer_bn(ochs[0], blkcnt[1], ochs[1], 1, strides[1],affine=affine,engine=engine);
    retlayers["2"] = make_body_layer_bn(ochs[1], blkcnt[2], ochs[2], 1, strides[2],affine=affine,engine=engine);
    retlayers["3"] = make_body_layer_bn(ochs[2], blkcnt[3], ochs[3], 1, strides[3],affine=affine,engine=engine);
    retlayers["4"] = make_body_layer_bn(ochs[3], blkcnt[4], ochs[4], 1, strides[4],affine=affine,engine=engine);
    retlayers["5"] = make_body_layer_bn(ochs[4], blkcnt[5], ochs[5], 1, strides[5],affine=affine,engine=engine);
    return retlayers;

# OSOCR config. Seems they have much better perf due to the heavier layout
# The called the method ``pami'' www

def res45p_bn(inpch,oupch,strides,frac=1,ochs=None,blkcnt=None,inplace=True,affine=True,engine=nn.BatchNorm2d):
    blkcnt = [None, 3, 4, 6, 6, 3];
    if ochs is None:
        ochs = [int(32*frac),int(64 * frac), int(128 * frac), int(256 * frac), int(512 * frac), oupch]
    retlayers = {};
    retlayers["0"] = make_init_layer_bn(ochs[0],affine=affine,engine=engine);
    retlayers["1"] = make_body_layer_bn(ochs[0], blkcnt[1], ochs[1], 1, strides[1],affine=affine,engine=engine);
    retlayers["2"] = make_body_layer_bn(ochs[1], blkcnt[2], ochs[2], 1, strides[2],affine=affine,engine=engine);
    retlayers["3"] = make_body_layer_bn(ochs[2], blkcnt[3], ochs[3], 1, strides[3],affine=affine,engine=engine);
    retlayers["4"] = make_body_layer_bn(ochs[3], blkcnt[4], ochs[4], 1, strides[4],affine=affine,engine=engine);
    retlayers["5"] = make_body_layer_bn(ochs[4], blkcnt[5], ochs[5], 1, strides[5],affine=affine,engine=engine);
    return retlayers;