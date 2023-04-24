
def getcmd(name,value):
    return "\\newcommand{" + name+"}{"+"{:.2f}".format(value*100)+"}";


def get_jpndictrec(codename,protocolname,dsname,branchname):
    dic={};
    dic["Accuracy"]="\\"+branchname+dsname+codename+protocolname+"la";
    dic["AR"]="\\"+branchname+dsname+codename+protocolname+"ca";
    dic["Recall"] = "\\" + branchname + dsname + codename + protocolname + "re";
    dic["Precision"] = "\\" + branchname + dsname + codename + protocolname + "pr";
    dic["Hmeans"] = "\\" + branchname + dsname + codename + protocolname + "hm";
    return dic;

def maketex(codename,protocolname,dsname,branchname,rec):
    dic=get_jpndictrec(codename,protocolname,dsname,branchname);
    lst=[]
    for k in rec:
        if(k in dic):
            lst.append(getcmd(dic[k],rec[k]).replace("-","at"))
    return lst;
