import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import cv2
import json
import editdistance as ed
import regex


def encode_static(label_batch, case_sensitive, tdict):
    max_len = max([len(regex.findall(r'\X', s, regex.U)) for s in label_batch])
    out = torch.zeros(len(label_batch), max_len + 1).long()
    for i in range(0, len(label_batch)):
        if not case_sensitive:
            cur_encoded = torch.tensor(
                [tdict.index(char.lower()) if char.lower() in tdict else len(tdict)
                 for char in regex.findall(r'\X', label_batch[i], regex.U)]) + 1
        else:
            cur_encoded = torch.tensor([tdict.index(char) if char in tdict else len(tdict)
                                        for char in regex.findall(r'\X', label_batch[i], regex.U)]) + 1
        out[i][0:len(cur_encoded)] = cur_encoded
    return out
    pass;
def decode_prob_static(net_out,length,tdict):

    # decoding prediction into text with geometric-mean probability
    # the probability is used to select the more realiable prediction when using bi-directional decoders
    out = []
    out_prob = []
    net_out = F.softmax(net_out, dim=1)
    for i in range(0, length.shape[0]):
        current_idx_list = net_out[int(length[:i].sum()): int(length[:i].sum() + length[i])].topk(1)[1][:, 0].tolist()
        current_text = ''.join([tdict[_ - 1] if _ > 0 and _ <= len(tdict) else '⑨' for _ in current_idx_list])
        current_probability = net_out[int(length[:i].sum()): int(length[:i].sum() + length[i])].topk(1)[0][:, 0]
        current_probability = torch.exp(torch.log(current_probability).sum() / current_probability.size()[0])
        out.append(current_text)
        out_prob.append(current_probability)
    return (out, out_prob)

def decode_raw_static(net_out,length,tdict):
    out = []
    out_prob = []
    out_raw = [];
    raw_out = net_out;

    net_out = F.softmax(net_out, dim=1)
    for i in range(0, length.shape[0]):
        current_idx_list = net_out[int(length[:i].sum()): int(length[:i].sum() + length[i])].topk(1)[1][:,
                           0].tolist()
        current_text = ''.join([tdict[_] if _ in tdict else '' for _ in current_idx_list])

        out_raw.append(list(
            raw_out[int(length[:i].sum()): int(length[:i].sum() + length[i])].topk(1)[0][:, 0].detach().cpu().numpy()));
        out.append(current_text)
    return (out, out_raw)
def decode_char_topk_static(net_out, length, protos, labels, tdict,topk_ch=5):
    net_out = F.softmax(net_out, dim=1);
    beams = [];
    idmat=[];
    for i in range(0, length.shape[0]):
        current_idx_list = net_out[int(length[:i].sum()): int(length[:i].sum() + length[i])].topk(topk_ch)[1].T.tolist();
        current_texts = [''.join([tdict[_] if _ in tdict else '~' for _ in current_idx_list]) for current_idx_list in
                         current_idx_list]
        idmat.append(current_idx_list);
        # if(current_texts[0]!=text_d[i]):
        #     print("Oops");
        beams.append(current_texts)
    return idmat,beams

def decode_word_topk_static(net_out, length, protos, labels, tdict,topk_ch=5,topk_beams=10):
    out = []
    out_prob = []
    out_raw = [];
    raw_out = net_out;
    net_out = F.softmax(net_out, dim=1)
    beams = [];
    for i in range(0, length.shape[0]):
        current_idx_list = net_out[int(length[:i].sum()): int(length[:i].sum() + length[i])].topk(topk_ch)[1].tolist()
        current_probability = net_out[int(length[:i].sum()): int(length[:i].sum() + length[i])].topk(topk_ch)[
            0].cpu().numpy();
        active_lists = [[]]
        active_prob_lists = [[]];
        active_plist = [1]
        for timestamp in range(int(length[i].item())):
            new_list = [];
            new_plist = [];
            new_prob_lists = [];
            aids = np.argsort(active_plist)[::-1][:topk_beams];
            for chid in range(topk_ch):
                pr = current_probability[timestamp][chid];
                ch = current_idx_list[timestamp][chid];
                for aid in aids:
                    new_list.append(active_lists[aid] + [ch]);
                    new_plist.append(pr * active_plist[aid]);
                    new_prob_lists.append(active_prob_lists[aid] + [pr])
            active_lists = new_list
            active_plist = new_plist;
            active_prob_lists = new_prob_lists;
            pass;
        aids = np.argsort(active_plist)[::-1][:topk_beams];
        current_idx_lists = [active_lists[aid] for aid in aids];
        current_probabilitys = [active_prob_lists[aid] for aid in aids];
        tdict[0] = "TOPNEP"
        current_texts = [''.join([tdict[_] if _ in tdict else '~' for _ in current_idx_list]) for current_idx_list in
                         current_idx_lists]
        current_texts = [t.split("TOPNEP")[0] for t in current_texts];

        # if(current_texts[0]!=text_d[i]):
        #     print("Oops");

        out_raw.append(list(raw_out[int(length[:i].sum()): int(length[:i].sum() + length[i])].topk(1)[0][:,
                            0].detach().cpu().numpy()));
        current_probability = 1  # torch.exp(torch.log(torch.tensor(current_probability)).sum() / current_probability.size[0])
        out.append(current_texts[0])
        out_prob.append(current_probability)
        beams.append(current_texts)
    return (out, out_raw, beams)

def decode_k_static(net_out,length,k,tdict):
    out = []
    net_out = F.softmax(net_out, dim=1)
    for i in range(0, length.shape[0]):
        current_idx_list = net_out[int(length[:i].sum()): int(length[:i].sum() + length[i])].topk(k)[1][:,
                           0].tolist()
        current_text = ''.join(
            [tdict[_ - 1] if _ > 0 and _ <= len(tdict) else '⑨' for _ in current_idx_list])
        out.append(current_text)
    return out
class cha_encdec():
    def __init__(this, dict_file, case_sensitive = True):
        this.dict = []
        this.case_sensitive = case_sensitive
        lines = open(dict_file , 'r').readlines()
        for line in lines:
            this.dict.append(line.replace('\n', ''))
    def encode(this, label_batch):
        return encode_static(label_batch,this.case_sensitive,this.dict);
    def decode(this, net_out, length):
        return decode_prob_static(net_out,length,this.dict);
    def decode_k(this,net_out,length,k=1):
        return decode_k_static(net_out,length,k,this.dict);




class neko_osfsl_ACR_counter():
    def __init__(this, display_string):
        this.correct = 0
        this.total_samples = 0.
        this.display_string = display_string
    def clear(this):
        this.correct = 0
        this.total_samples = 0.

    def add_iter(this,pred, labels):
        cnt=len(labels);
        this.total_samples +=cnt;
        for i in range(cnt):
            if(pred[i]==labels[i]):
                this.correct+=1;

    def show(this):
        # Accuracy for scene text.
        # CER and WER for handwritten text.
        print(this.display_string)
        if this.total_samples == 0:
            pass
        print('Accuracy: {:.6f}'.format(
            this.correct / this.total_samples,))

class neko_os_ACR_counter():
    def __init__(this, display_string):
        this.correct = 0
        this.total_samples = 0.
        this.display_string = display_string
    def clear(this):
        this.correct = 0
        this.total_samples = 0.

    def add_iter(this,pred, labels):
        this.total_samples += labels.shape[0]
        this.correct+=(labels==pred).sum().item();

    def show(this):
        # Accuracy for scene text.
        # CER and WER for handwritten text.
        print(this.display_string)
        if this.total_samples == 0:
            pass
        print('Accuracy: {:.6f}'.format(
            this.correct / this.total_samples,))

class neko_os_Attention_AR_counter():
    def __init__(this, display_string, case_sensitive):
        this.correct = 0
        this.total_samples = 0.
        this.distance_C = 0
        this.total_C = 0.
        this.distance_W = 0
        this.total_W = 0.
        this.display_string = display_string
        this.case_sensitive = case_sensitive

    def clear(this):
        this.correct = 0
        this.total_samples = 0.
        this.distance_C = 0
        this.total_C = 0.
        this.distance_W = 0
        this.total_W = 0.

    def add_iter(this,prdt_texts, label_length, labels,debug=False):
        if(labels is None):
            return ;
        start = 0
        start_o = 0
        if debug:
            for i in range(len(prdt_texts)):
                print(labels[i], "->-", prdt_texts[i]);

        this.total_samples += len(labels);
        for i in range(0, len(prdt_texts)):
            if not this.case_sensitive:
                prdt_texts[i] = prdt_texts[i].lower().replace("⑨","")
                labels[i] = labels[i].lower()
            all_words = []
            for w in labels[i].split('|sadhkjashfkjasyhf') + prdt_texts[i].split('||sadhkjashfkjasyhf'):
                if w not in all_words:
                    all_words.append(w)
            l_words = [all_words.index(_) for _ in labels[i].split('||sadhkjashfkjasyhf')]
            p_words = [all_words.index(_) for _ in prdt_texts[i].split('||sadhkjashfkjasyhf')]
            this.distance_C += ed.eval(labels[i], prdt_texts[i])
            this.distance_W += ed.eval(l_words, p_words)
            this.total_C += len(labels[i])
            this.total_W += len(l_words)
            this.correct = this.correct + 1 if labels[i] == prdt_texts[i] else this.correct

    def show(this):
        # Accuracy for scene text.
        # CER and WER for handwritten text.
        print(this.display_string)
        if this.total_samples == 0:
            pass
        astr='Accuracy: {:.6f}, AR: {:.6f}, CER: {:.6f}, WER: {:.6f}'.format(
            this.correct / this.total_samples,
            1 - this.distance_C / this.total_C,
            this.distance_C / this.total_C,
            this.distance_W / this.total_W);
        print(astr)
        print("Total Samples:",this.total_samples)
        return {
            "Accuracy": this.correct / this.total_samples,
            "AR": 1 - this.distance_C / this.total_C,
            "Total Samples" : this.total_samples
        }
class neko_oswr_Attention_AR_counter():
    def __init__(this, display_string, case_sensitive):
        this.clear()
        this.display_string = display_string
        this.case_sensitive = case_sensitive

    def clear(this):
        this.correct = 0
        this.total_samples = 0.
        this.total_C = 0.
        this.total_W = 0.
        this.total_U=0.
        this.total_K=0.
        this.Ucorr=0.
        this.Kcorr=0.
        this.KtU=0.

    def add_iter(this,prdt_texts, label_length, labels,debug=False,UNK="⑨"):
        if(labels is None):
            return ;
        start = 0
        start_o = 0
        if debug:
            for i in range(len(prdt_texts)):
                print(labels[i], "->-", prdt_texts[i]);

        this.total_samples += len(labels);
        for i in range(0, len(prdt_texts)):
            if not this.case_sensitive:
                prdt_texts[i] = prdt_texts[i].lower()
                labels[i] = labels[i].lower()
            all_words = []
            for w in labels[i].split('|sadhkjashfkjasyhf') + prdt_texts[i].split('||sadhkjashfkjasyhf'):
                if w not in all_words:
                    all_words.append(w)
            l_words = [all_words.index(_) for _ in labels[i].split('||sadhkjashfkjasyhf')]
            p_words = [all_words.index(_) for _ in prdt_texts[i].split('||sadhkjashfkjasyhf')]
            this.total_C += len(labels[i])
            this.total_W += len(l_words)
            cflag=int(labels[i] == prdt_texts[i]);
            this.correct = this.correct + cflag;
            if(labels[i].find(UNK)!=-1):
                this.total_U+=1;
                this.Ucorr+=(prdt_texts[i].find(UNK)!=-1);
            else:
                this.total_K+=1;
                this.Kcorr+=cflag;
                this.KtU+=(prdt_texts[i].find(UNK)!=-1);


    def show(this):
        # Accuracy for scene text.
        # CER and WER for handwritten text.
        print(this.display_string)
        if this.total_samples == 0:
            pass
        R=this.Ucorr / max(this.total_U,1);
        P=this.Ucorr / max(this.Ucorr + this.KtU,1);
        F=2*(R*P)/max(R+P,1.)

        print(' KACR: {:.6f},URCL:{:.6f}, UPRE {:.6f}, F {:.6f}'.format(
            this.Kcorr / max(1,this.total_K),
            R,P,F
        ))
        return {"Accuracy" :this.Kcorr / max(this.total_K,1), "Recall":R, "Precision": P,"Hmeans":F}

class Attention_AR_counter():
    def __init__(this, display_string, dict_file, case_sensitive):
        this.correct = 0
        this.total_samples = 0.
        this.distance_C = 0
        this.total_C = 0.
        this.distance_W = 0
        this.total_W = 0.
        this.display_string = display_string
        this.case_sensitive = case_sensitive
        this.de = cha_encdec(dict_file, case_sensitive)

    def clear(this):
        this.correct = 0
        this.total_samples = 0.
        this.distance_C = 0
        this.total_C = 0.
        this.distance_W = 0
        this.total_W = 0.
        
    def add_iter(this, output, out_length, label_length, labels,debug=False):
        this.total_samples += label_length.size()[0]
        raw_prdts = output.topk(1)[1]
        prdt_texts, prdt_prob = this.de.decode(output, out_length)
        CS=[]
        DS=[];
        batch_corr=0;
        batch_tot=0;
        if debug:
            for i in range(len(prdt_texts)):
                print(labels[i].lower(),"->-",prdt_texts[i].lower());
        prdt_texts=[i.replace("⑨","") for i in prdt_texts];
        for i in range(0, len(prdt_texts)):
            if not this.case_sensitive:
                prdt_texts[i] = prdt_texts[i].lower()
                labels[i] = labels[i].lower()
            all_words = []
            for w in labels[i].split('|') + prdt_texts[i].split('|'):
                if w not in all_words:
                    all_words.append(w)
            l_words = [all_words.index(_) for _ in labels[i].split('|')]
            p_words = [all_words.index(_) for _ in prdt_texts[i].split('|')]
            this.distance_C += ed.eval(labels[i], prdt_texts[i])
            this.distance_W += ed.eval(l_words, p_words)
            this.total_C += len(labels[i])
            this.total_W += len(l_words)
            CS.append(len(labels[i]));
            DS.append(ed.eval(labels[i], prdt_texts[i]));
            this.correct = this.correct + 1 if labels[i] == prdt_texts[i] else this.correct
            batch_corr = batch_corr + 1 if labels[i] == prdt_texts[i] else batch_corr;
            batch_tot+=1;
        return batch_corr/batch_tot,CS,DS;
    def show(this):
    # Accuracy for scene text. 
    # CER and WER for handwritten text.
        print(this.display_string)
        if this.total_samples == 0:
            pass
        print('Accuracy: {:.6f}, AR: {:.6f}, CER: {:.6f}, WER: {:.6f}'.format(
            this.correct / this.total_samples,
            1 - this.distance_C / this.total_C,
            this.distance_C / this.total_C,
            this.distance_W / this.total_W))


class Attention_AR_counter_node():
    def __init__(this, display_string, case_sensitive):
        this.correct = 0
        this.total_samples = 0.
        this.distance_C = 0
        this.total_C = 0.
        this.distance_W = 0
        this.total_W = 0.
        this.display_string = display_string
        this.case_sensitive = case_sensitive

    def clear(this):
        this.correct = 0
        this.total_samples = 0.
        this.distance_C = 0
        this.total_C = 0.
        this.distance_W = 0
        this.total_W = 0.

    def add_iter(this, output, out_length, label_length, labels,de, debug=False):
        this.total_samples += label_length.size()[0]
        raw_prdts = output.topk(1)[1]
        prdt_texts, prdt_prob = de.decode(output, out_length)
        if debug:
            for i in range(len(prdt_texts)):
                print(labels[i].lower(), "->-", prdt_texts[i].lower());
        prdt_texts = [i.replace("⑨", "") for i in prdt_texts];
        for i in range(0, len(prdt_texts)):
            if not this.case_sensitive:
                prdt_texts[i] = prdt_texts[i].lower()
                labels[i] = labels[i].lower()
            all_words = []
            for w in labels[i].split('|') + prdt_texts[i].split('|'):
                if w not in all_words:
                    all_words.append(w)
            l_words = [all_words.index(_) for _ in labels[i].split('|')]
            p_words = [all_words.index(_) for _ in prdt_texts[i].split('|')]
            this.distance_C += ed.eval(labels[i], prdt_texts[i])
            this.distance_W += ed.eval(l_words, p_words)
            this.total_C += len(labels[i])
            this.total_W += len(l_words)
            this.correct = this.correct + 1 if labels[i] == prdt_texts[i] else this.correct

    def show(this):
        # Accuracy for scene text.
        # CER and WER for handwritten text.
        print(this.display_string)
        if this.total_samples == 0:
            pass
        print('Accuracy: {:.6f}, AR: {:.6f}, CER: {:.6f}, WER: {:.6f}'.format(
            this.correct / this.total_samples,
            1 - this.distance_C / this.total_C,
            this.distance_C / this.total_C,
            this.distance_W / this.total_W))

class Loss_counter():
    def __init__(this, display_interval):
        this.display_interval = display_interval
        this.total_iters = 0.
        this.loss_sum = 0
        this.termsum={};
    def mkterm(this,dic,prfx=""):
        for k in dic:
            if(type(dic[k])==dict):
                this.mkterm(dic[k],prfx+k)
            else:
                if prfx+k not in this.termsum:
                    this.termsum[prfx+k] = 0;
                this.termsum[prfx+k] += float(dic[k])

    def add_iter(this, loss,terms=None):
        this.total_iters += 1
        this.loss_sum += float(loss)
        if terms is not None:
            this.mkterm(terms);

    def clear(this):
        this.total_iters = 0
        this.loss_sum = 0
        this.termsum={};
    def show(this):
        terms=this.get_loss_and_terms();
        print(terms);
        return terms;
    def get_loss(this):
        loss = this.loss_sum / this.total_iters if this.total_iters > 0 else 0
        this.clear();
        return loss
    def get_loss_and_terms(this):
        loss = this.loss_sum / this.total_iters if this.total_iters > 0 else 0
        retterms={};
        for k in this.termsum:
            term = this.termsum[k] / this.total_iters if this.total_iters > 0 else 0;
            retterms[k]=term;
        this.clear();
        return loss,retterms;

