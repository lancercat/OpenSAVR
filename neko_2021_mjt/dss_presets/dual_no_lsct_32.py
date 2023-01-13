from neko_2021_mjt.dataloaders.neko_joint_loader import neko_joint_loader
from neko_2021_mjt.configs.data.mjst_data import get_mjstcqa_cfg,get_test_all_uncased_dsrgb,get_test_all_uncased_dsrgb_all
from neko_2021_mjt.configs.data.chs_jap_data import get_chs_HScqa,get_eval_jap_color,\
    get_chs_tr_meta,get_chs_sc_meta,get_chs_mc_meta,\
    get_jap_te_meta,get_eval_monkey_color,\
    get_jap_te_meta_gosr,get_jap_te_meta_ostr,get_jap_te_meta_osr,\
    get_eval_kr_color,get_kr_te_meta, \
    get_eval_be_color,get_be_te_meta, \
    get_eval_hn_color,get_hn_te_meta
import os

def get_dataloadercfgs(root,te_meta_path,tr_meta_path,maxT_mjst,maxT_chsHS,bsize,devices=None):
    dss= {
        "loadertype": neko_joint_loader,
        "subsets":
        {
        }
    };
    if(maxT_mjst>0):
        # reset iterator gives deadlock, so we give a large enough repeat number
        dss["subsets"]["dan_mjst"]= get_mjstcqa_cfg(root, maxT_mjst, bs=bsize, hw=[32, 128]);
    if(maxT_chsHS>0):
        dss["subsets"]["dan_chs"] =  get_chs_HScqa(root, maxT_chsHS, bsize, -1, hw=[32, 128]);
    return dss;

def get_dataloadercfgsch(root,te_meta_path,tr_meta_path,maxT_mjst,maxT_chsHS,bsize):
    return \
    {
        "loadertype": neko_joint_loader,
        "subsets":
        {
            # reset iterator gives deadlock, so we give a large enough repeat number
            # "dan_mjst":get_mjstcqa_cfg(root, maxT_mjst, bs=bsize,hw=[32,128]),
            "dan_chs": get_chs_HScqa(root, maxT_chsHS, bsize, -1,hw=[32,128]),
        }
    }
def get_dataloadercfgsms(root,te_meta_path,tr_meta_path,maxT_mjst,maxT_chsHS,bsize):
    return \
    {
        "loadertype": neko_joint_loader,
        "subsets":
        {
            # reset iterator gives deadlock, so we give a large enough repeat number
            "dan_mjst":get_mjstcqa_cfg(root, maxT_mjst, bs=bsize,hw=[32,128]),
            #"dan_chs": get_chs_HScqa(root, maxT_chsHS, bsize, -1,hw=[32,128]),
        }
    }
def get_eval_dss(dsroot,maxT_mjst,maxT_chs,batch_size=1,lhw=[32,128]):
    te_meta_path_chsjap = get_jap_te_meta(dsroot);
    te_meta_path_mjst = os.path.join(dsroot, "dicts", "dab62cased.pt");
    mjst_eval_ds = get_test_all_uncased_dsrgb(maxT_mjst, dsroot, None,hw=lhw,batchsize=batch_size)
    chs_eval_ds = get_eval_jap_color(dsroot, maxT_chs,hw=lhw);
    return te_meta_path_chsjap,te_meta_path_mjst,mjst_eval_ds,chs_eval_ds

def get_eval_dss_gzsl(dsroot,maxT_mjst,maxT_chs,batch_size=1,lhw=[32,128]):
    te_meta_path_chsjap = get_jap_te_meta(dsroot);
    chs_eval_ds = get_eval_jap_color(dsroot, maxT_chs,hw=lhw);
    return te_meta_path_chsjap,chs_eval_ds,False



def get_eval_dss_gosr(dsroot,maxT_mjst,maxT_chs,lhw=[32,128]):
    te_meta_path_chsjap = get_jap_te_meta_gosr(dsroot);
    chs_eval_ds = get_eval_jap_color(dsroot, maxT_chs,hw=lhw);
    return te_meta_path_chsjap,chs_eval_ds,True
def get_eval_dss_osr(dsroot, maxT_mjst, maxT_chs,lhw=[32,128]):
    te_meta_path_chsjap = get_jap_te_meta_osr(dsroot);
    chs_eval_ds = get_eval_jap_color(dsroot, maxT_chs,hw=lhw);
    return te_meta_path_chsjap,chs_eval_ds,True
def get_eval_dss_ostr(dsroot, maxT_mjst, maxT_chs,lhw=[32,128]):
    te_meta_path_chsjap = get_jap_te_meta_ostr(dsroot);
    chs_eval_ds = get_eval_jap_color(dsroot, maxT_chs,hw=lhw);
    return te_meta_path_chsjap,chs_eval_ds,True

def get_bench_dss_all(dsroot,maxT_mjst,maxT_chs,batch_size=1):
    te_meta_path_chsjap = get_jap_te_meta(dsroot);
    te_meta_path_mjst = os.path.join(dsroot, "dicts", "dab62cased.pt");
    mjst_eval_ds = get_test_all_uncased_dsrgb_all(maxT_mjst, dsroot, None, batch_size,hw=[32,128])
    chs_eval_ds = get_eval_jap_color(dsroot, maxT_chs,hw=[32,128]);
    return te_meta_path_chsjap,te_meta_path_mjst,mjst_eval_ds,chs_eval_ds

def get_bench_dss(dsroot,maxT_mjst,maxT_chs,batch_size=1):
    te_meta_path_chsjap = get_jap_te_meta(dsroot);
    te_meta_path_mjst = os.path.join(dsroot, "dicts", "dab62cased.pt");
    mjst_eval_ds = get_test_all_uncased_dsrgb(maxT_mjst, dsroot, None, batch_size,hw=[32,128])
    chs_eval_ds = get_eval_jap_color(dsroot, maxT_chs,hw=[32,128]);
    return te_meta_path_chsjap,te_meta_path_mjst,mjst_eval_ds,chs_eval_ds
def get_eval_dss_kr(dsroot,maxT_mjst,maxT_chs):
    te_meta_path_chsjap = get_kr_te_meta(dsroot);
    te_meta_path_mjst = os.path.join(dsroot, "dicts", "dab62cased.pt");
    mjst_eval_ds = get_test_all_uncased_dsrgb(maxT_mjst, dsroot, None,hw=[32,128])
    chs_eval_ds = get_eval_kr_color(dsroot, maxT_chs,hw=[32,128]);

    return te_meta_path_chsjap,te_meta_path_mjst,mjst_eval_ds,chs_eval_ds

def get_eval_dss_ba(dsroot,maxT_mjst,maxT_chs):
    te_meta_path_chsjap = get_be_te_meta(dsroot);
    te_meta_path_mjst = os.path.join(dsroot, "dicts", "dab62cased.pt");
    mjst_eval_ds = get_test_all_uncased_dsrgb(maxT_mjst, dsroot, None, 16,hw=[32,128])
    chs_eval_ds = get_eval_be_color(dsroot, maxT_chs,hw=[32,128]);
    return te_meta_path_chsjap,te_meta_path_mjst,mjst_eval_ds,chs_eval_ds

def get_eval_dss_hn(dsroot,maxT_mjst,maxT_chs):
    te_meta_path_chsjap = get_hn_te_meta(dsroot);
    te_meta_path_mjst = os.path.join(dsroot, "dicts", "dab62cased.pt");
    mjst_eval_ds = get_test_all_uncased_dsrgb(maxT_mjst, dsroot, None, 16,hw=[32,128])
    chs_eval_ds = get_eval_hn_color(dsroot, maxT_chs,hw=[32,128]);
    return te_meta_path_chsjap,te_meta_path_mjst,mjst_eval_ds,chs_eval_ds


def get_dss(dsroot,maxT_mjst,maxT_chs,bsize):
    te_meta_path_chsjap, te_meta_path_mjst, mjst_eval_ds, chs_eval_ds=\
        get_eval_dss(dsroot,maxT_mjst,maxT_chs);
    tr_meta_path_chsjap = get_chs_tr_meta(dsroot);
    tr_meta_path_mjst = os.path.join(dsroot, "dicts", "dab62cased.pt");
    train_joint_ds=get_dataloadercfgs(dsroot,te_meta_path_mjst,tr_meta_path_chsjap,maxT_mjst,maxT_chs,bsize);
    return tr_meta_path_chsjap,te_meta_path_chsjap,tr_meta_path_mjst,te_meta_path_mjst,mjst_eval_ds,chs_eval_ds,train_joint_ds


def get_eval_dss_m(dsroot,maxT_mjst,maxT_chs,lang="chs"):
    te_meta_path_chsjap = get_jap_te_meta(dsroot);
    te_meta_path_mjst = os.path.join(dsroot, "dicts", "dab62cased.pt");
    mjst_eval_ds = get_test_all_uncased_dsrgb(maxT_mjst, dsroot, None, 16,hw=[32,128])
    japm_eval_ds = get_eval_monkey_color(dsroot, maxT_chs,lang,hw=[32,128]);

    return te_meta_path_chsjap,te_meta_path_mjst,mjst_eval_ds,japm_eval_ds



def get_dsssc(dsroot,maxT_mjst,maxT_chs,bsize):
    te_meta_path_chsjap, te_meta_path_mjst, mjst_eval_ds, chs_eval_ds=\
        get_eval_dss(dsroot,maxT_mjst,maxT_chs);
    tr_meta_path_chsjap = get_chs_sc_meta(dsroot);
    tr_meta_path_mjst = os.path.join(dsroot, "dicts", "dab62cased.pt");
    train_joint_ds=get_dataloadercfgs(dsroot,te_meta_path_mjst,tr_meta_path_chsjap,maxT_mjst,maxT_chs,bsize);
    return tr_meta_path_chsjap,te_meta_path_chsjap,tr_meta_path_mjst,te_meta_path_mjst,mjst_eval_ds,chs_eval_ds,train_joint_ds

def get_dssmc(dsroot,maxT_mjst,maxT_chs,bsize):
    te_meta_path_chsjap, te_meta_path_mjst, mjst_eval_ds, chs_eval_ds=\
        get_eval_dss(dsroot,maxT_mjst,maxT_chs);
    tr_meta_path_chsjap = get_chs_mc_meta(dsroot);
    tr_meta_path_mjst = os.path.join(dsroot, "dicts", "dab62cased.pt");
    train_joint_ds=get_dataloadercfgs(dsroot,te_meta_path_mjst,tr_meta_path_chsjap,maxT_mjst,maxT_chs,bsize);
    return tr_meta_path_chsjap,te_meta_path_chsjap,tr_meta_path_mjst,te_meta_path_mjst,mjst_eval_ds,chs_eval_ds,train_joint_ds
def get_dsssch(dsroot,maxT_mjst,maxT_chs,bsize):
    te_meta_path_chsjap, te_meta_path_mjst, mjst_eval_ds, chs_eval_ds=\
        get_eval_dss(dsroot,maxT_mjst,maxT_chs);
    tr_meta_path_chsjap = get_chs_sc_meta(dsroot);
    tr_meta_path_mjst = None;
    train_joint_ds=get_dataloadercfgsch(dsroot,te_meta_path_mjst,tr_meta_path_chsjap,maxT_mjst,maxT_chs,bsize);
    return tr_meta_path_chsjap,te_meta_path_chsjap,tr_meta_path_mjst,te_meta_path_mjst,mjst_eval_ds,chs_eval_ds,train_joint_ds

def get_dssscm(dsroot,maxT_mjst,maxT_chs,bsize):
    te_meta_path_chsjap, te_meta_path_mjst, mjst_eval_ds, chs_eval_ds=\
        get_eval_dss(dsroot,maxT_mjst,maxT_chs);
    tr_meta_path_chsjap =None;
    tr_meta_path_mjst = os.path.join(dsroot, "dicts", "dab62cased.pt");
    train_joint_ds=get_dataloadercfgsms(dsroot,te_meta_path_mjst,tr_meta_path_chsjap,maxT_mjst,maxT_chs,bsize);
    return tr_meta_path_chsjap,te_meta_path_chsjap,tr_meta_path_mjst,te_meta_path_mjst,mjst_eval_ds,chs_eval_ds,train_joint_ds



def get_dssscht(dsroot,maxT_chs,bsize):
    tr_meta_path_chsjap = get_chs_sc_meta(dsroot);
    train_joint_ds=get_dataloadercfgsch(dsroot,None,None,None,maxT_chs,bsize);
    return tr_meta_path_chsjap,train_joint_ds

def get_eval_dss_jk(dsroot,maxT_chs,batch_size=1):
    te_meta_path_jap = get_jap_te_meta(dsroot);
    te_meta_path_kr = get_kr_te_meta(dsroot);
    jap_eval_ds = get_eval_jap_color(dsroot, maxT_chs,hw=[32,128]);
    kr_eval_ds = get_eval_kr_color(dsroot, maxT_chs,hw=[32,128]);
    return te_meta_path_jap,te_meta_path_kr,jap_eval_ds,kr_eval_ds

def get_eval_dss_close(dsroot,maxT_mjst,batch_size=1):
    te_meta_path_mjst = os.path.join(dsroot, "dicts", "dab62cased.pt");
    mjst_eval_ds = get_test_all_uncased_dsrgb(maxT_mjst, dsroot, None,hw=[32,128],batchsize=batch_size)
    return te_meta_path_mjst,mjst_eval_ds



def get_dss_close(dsroot,maxT_mjst,bsize):
    te_meta_path_mjst, mjst_eval_ds=get_eval_dss_close(dsroot,maxT_mjst)
    tr_meta_path_mjst = os.path.join(dsroot, "dicts", "dab62cased.pt");
    train_joint_ds=get_dataloadercfgs(dsroot,te_meta_path_mjst,None,maxT_mjst,-1,bsize);
    return tr_meta_path_mjst,te_meta_path_mjst,mjst_eval_ds,train_joint_ds
