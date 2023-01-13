# coding:utf-8
import random
from torch.utils.data import Dataset
import lmdb
import six
import sys
from PIL import Image
import numpy as np
from neko_sdk.ocr_modules.io.data_tiding import neko_DAN_padding
class lmdbDataset(Dataset):
    def init_etc(this):
        pass;
    def set_dss(self,roots):
        for i in range(0, len(roots)):
            self.root_paths.append(roots[i])
            env = lmdb.open(
                roots[i],
                max_readers=1,
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False)
            if not env:
                print('cannot creat lmdb from %s' % (roots[i]))
                sys.exit(0)
            with env.begin(write=False) as txn:
                nSamples = int(txn.get('num-samples'.encode()))
                self.nSamples += nSamples
            self.lengths.append(nSamples)
            self.roots.append(roots[i]);
            self.envs.append(env);
            self.init_etc();

    def __init__(self, roots=None, ratio=None, img_height = 32, img_width = 128,
        transform=None, global_state='Test',maxT=25,repeat=1,qhb_aug=False,force_target_ratio=None,novert=True):
        self.envs = []
        self.roots=[];
        self.root_paths=[];
        self.maxT=maxT;
        self.nSamples = 0
        self.lengths = []
        self.ratio = []
        self.global_state = global_state
        self.repeat=repeat;
        self.qhb_aug=qhb_aug;
        self.set_dss(roots)
        self.novert=novert;
        if ratio != None:
            assert len(roots) == len(ratio) ,'length of ratio must equal to length of roots!'
            for i in range(0,len(roots)):
                self.ratio.append(ratio[i] / float(sum(ratio)))
        else:
            for i in range(0,len(roots)):
                self.ratio.append(self.lengths[i] / float(self.nSamples))

        self.transform = transform
        self.maxlen = max(self.lengths)
        self.img_height = img_height
        self.img_width = img_width
        # Issue12
        if(force_target_ratio is None):
            try:
                self.target_ratio = img_width / float(img_height)
            except:
                print("failed setting target_ration")
        else:
            self.target_ratio = force_target_ratio
        # self.statistic();

    def statistic(this):
        from neko_sdk.ocr_modules.charset.jap_cset import hira,kata;
        from neko_sdk.ocr_modules.charset.chs_cset import t1_3755;
        from neko_sdk.ocr_modules.charset.etc_cset import latin54, digits;

        print(this.nSamples);
        total_dict = {
            "total": 0,
            "Kana":0,
            "digit": 0,
            "t1_chinese": 0,
            "latin54": 0,
            "other":0,
        }
        total_l_dict = {
            "total": 0,
            "Kana":0,
            "digit": 0,
            "t1_chinese": 0,
            "latin54": 0,
            "other": 0,
        }
        for i in range(0, len(this.roots)):
            print("path", this.root_paths[i]);
            print("length", this.lengths[i]);
            char_dict = {
                "total": 0,
                "Kana": 0,
                "digit": 0,
                "t1_chinese": 0,
                "latin54": 0,
                "other":0,
            }
            line_dict = {
            "total": 0,
            "Kana":0,
            "digit": 0,
            "t1_chinese": 0,
            "latin54": 0,
            "other":0,
            }
            if (this.lengths[i] > 160626):
                continue;
            for j in range(this.lengths[i]):
                with this.envs[i].begin(write=False) as txn:
                    label_key = 'label-%09d' % j;
                    try:
                        label = str(txn.get(label_key.encode()).decode('utf-8'));
                    except:
                        print("corrupt", i, j);
                        continue;
                    has_t1ch=0;
                    has_lat = 0;
                    has_dig=0;
                    has_kana=0;
                    has_other=0;
                    for c in label:
                        char_dict["total"] += 1;
                        total_dict["total"] += 1;
                        if (c in t1_3755):
                            char_dict["t1_chinese"] += 1;
                            total_dict["t1_chinese"] += 1;
                            has_t1ch = 1
                        elif (c in latin54):
                            char_dict["latin54"] += 1;
                            total_dict["latin54"] += 1;
                            has_lat=1
                        elif (c in digits):
                            char_dict["digit"] += 1;
                            total_dict["digit"] += 1;
                            has_dig=1;
                        elif (c in hira or c in kata):
                            char_dict["Kana"] += 1;
                            total_dict["Kana"] += 1;
                            has_kana=1;
                        else:
                            char_dict["other"] += 1;
                            total_dict["other"] += 1;
                            has_other=1;
                            print(c);
                line_dict["total"] += 1;
                line_dict["digit"]+=has_dig;
                line_dict["t1_chinese"] += has_t1ch;
                line_dict["latin54"] += has_lat;
                line_dict["other"] += has_other;
                line_dict["Kana"] += has_kana;

                total_l_dict["total"] += 1;
                total_l_dict["digit"] += has_dig;
                total_l_dict["t1_chinese"] += has_t1ch;
                total_l_dict["latin54"] += has_lat;
                total_l_dict["other"] += has_other;
                total_l_dict["Kana"] += has_kana;

            print("chcnt", char_dict);
            print("lncnt", line_dict);


        print("totlncnt", total_l_dict);
        print("totcnt", total_dict);


    # I dunno---- why would anyone want to know exactly how large the set is?

    def __fromwhich__(self ):
        rd = random.random()
        total = 0
        for i in range(0,len(self.ratio)):
            total += self.ratio[i]
            if rd <= total:
                return i

    def keepratio_resize(self, img):
        img,bmask=neko_DAN_padding(img,None,
                                       img_width=self.img_width,
                                       img_height=self.img_height,
                                       target_ratio=self.target_ratio,
                                       qhb_aug=self.qhb_aug,gray=True)
        return img,bmask

    def __len__(self):
        return self.nSamples
    def __getitem__(self, index):
        fromwhich = self.__fromwhich__()
        if self.global_state == 'Train':
            index = random.randint(0,self.maxlen - 1)
        index = index % self.lengths[fromwhich]
        assert index <= len(self), 'index range error'
        index += 1
        with self.envs[fromwhich].begin(write=False) as txn:
            img_key = 'image-%09d' % index
            try:
                imgbuf = txn.get(img_key.encode())
                buf = six.BytesIO()
                buf.write(imgbuf)
                buf.seek(0)
                img = Image.open(buf)
                label_key = 'label-%09d' % index
                label = str(txn.get(label_key.encode()).decode('utf-8'));
            except:
                print('Corrupted image for %d' % index)
                return self[index + 1]

            # if len(label) > 2 and img.width*2 < img.height:
            #     print('vertical',label,img.width /img.height)
            #     return self[index + 1]

            if len(label) > self.maxT-1 and self.global_state == 'Train':
                print('sample too long')
                return self[index + 1]
            try:
                img,bmask = self.keepratio_resize(img.convert('RGB'))
            except:
                print('Size error for %d' % index)
                return self[index + 1]
            if(len(img.shape)==2):
                img = img[:,:,np.newaxis]
            if self.transform:
                img = self.transform(img)
                bmask = self.transform(bmask)

            sample = {'image': img, 'label': label,"bmask": bmask}
            return sample




class colored_lmdbDataset(lmdbDataset):
    def keepratio_resize(self, img):
        img,bmask=neko_DAN_padding(img,None,
                                       img_width=self.img_width,
                                       img_height=self.img_height,
                                       target_ratio=self.target_ratio,
                                       qhb_aug=self.qhb_aug,gray=False)
        return img,bmask;


class colored_lmdbDatasetT(colored_lmdbDataset):
    pass;

class colored_lmdbDataset_semi(colored_lmdbDataset):
    def __init__(this, roots=None,cased_annoatations=None, ratio=None, img_height = 32, img_width = 128,
        transform=None, global_state='Test',maxT=25,repeat=1,qhb_aug=False):
        super(colored_lmdbDataset_semi, this).__init__(roots,ratio,img_height,img_width,transform,global_state,maxT,repeat,qhb_aug);
        this.cased=cased_annoatations;

    def __getitem__(self, index):
        fromwhich = self.__fromwhich__()
        if self.global_state == 'Train':
            index = random.randint(0, self.maxlen - 1)
        index = index % self.lengths[fromwhich]
        assert index <= len(self), 'index range error'
        index += 1
        with self.envs[fromwhich].begin(write=False) as txn:
            img_key = 'image-%09d' % index
            try:
                imgbuf = txn.get(img_key.encode())
                buf = six.BytesIO()
                buf.write(imgbuf)
                buf.seek(0)
                img = Image.open(buf)

            except:
                print('Corrupted image for %d' % index)
                return self[index + 1]
            label_key = 'label-%09d' % index
            label = str(txn.get(label_key.encode()).decode('utf-8'));
            # if len(label) > 2 and img.width*2 < img.height:
            #     print('vertical',label,img.width /img.height)
            #     return self[index + 1]

            if len(label) > self.maxT - 1 and self.global_state == 'Train':
                print('sample too long')
                return self[index + 1]
            try:
                img, bmask = self.keepratio_resize(img.convert('RGB'))
            except:
                print('Size error for %d' % index)
                return self[index + 1]
            if (len(img.shape) == 2):
                img = img[:, :, np.newaxis]
            if self.transform:
                img = self.transform(img)
                bmask = self.transform(bmask)

            sample = {'image': img, 'label': label,"cased": self.cased[fromwhich]}
            return sample

class lmdbDataset_semi(lmdbDataset):
    def __init__(this, roots=None,cased_annoatations=None, ratio=None, img_height = 32, img_width = 128,
        transform=None, global_state='Test',maxT=25,repeat=1,qhb_aug=False):
        super(lmdbDataset_semi, this).__init__(roots,ratio,img_height,img_width,transform,global_state,maxT,repeat,qhb_aug);
        this.cased=cased_annoatations;

    def __getitem__(self, index):
        fromwhich = self.__fromwhich__()
        if self.global_state == 'Train':
            index = random.randint(0, self.maxlen - 1)
        index = index % self.lengths[fromwhich]
        assert index <= len(self), 'index range error'
        index += 1
        with self.envs[fromwhich].begin(write=False) as txn:
            img_key = 'image-%09d' % index
            try:
                imgbuf = txn.get(img_key.encode())
                buf = six.BytesIO()
                buf.write(imgbuf)
                buf.seek(0)
                img = Image.open(buf)

            except:
                print('Corrupted image for %d' % index)
                return self[index + 1]
            label_key = 'label-%09d' % index
            label = str(txn.get(label_key.encode()).decode('utf-8'));
            # if len(label) > 2 and img.width*2 < img.height:
            #     print('vertical',label,img.width /img.height)
            #     return self[index + 1]

            if len(label) > self.maxT - 1 and self.global_state == 'Train':
                print('sample too long')
                return self[index + 1]
            try:
                img, bmask = self.keepratio_resize(img.convert('RGB'))
            except:
                print('Size error for %d' % index)
                return self[index + 1]
            if (len(img.shape) == 2):
                img = img[:, :, np.newaxis]
            if self.transform:
                img = self.transform(img)
                bmask = self.transform(bmask)

            sample = {'image': img, 'label': label,"cased": self.cased[fromwhich]}
            return sample



# some chs datasets are too smol.
class lmdbDataset_repeat(lmdbDataset):
    def __len__(self):
        return self.nSamples*self.repeat;

    def __getitem__(self, index):
        index%=self.nSamples;
        fromwhich = self.__fromwhich__()
        if self.global_state == 'Train':
            index = random.randint(0,self.maxlen - 1)
        index = index % self.lengths[fromwhich]
        assert index <= len(self), 'index range error'
        index += 1
        with self.envs[fromwhich].begin(write=False) as txn:
            img_key = 'image-%09d' % index
            try:
                imgbuf = txn.get(img_key.encode())
                buf = six.BytesIO()
                buf.write(imgbuf)
                buf.seek(0)
                img = Image.open(buf)
            except:
                print('Corrupted image for %d' % index)
                print("reponame: " ,self.roots[fromwhich]);
                return self[index + 1]

            label_key = 'label-%09d' % index
            try:
                label = str(txn.get(label_key.encode()).decode('utf-8'));
            except:
                print("reponame: " ,self.roots[fromwhich]);
            if len(label) > self.maxT-1 and self.global_state == 'Train':
                print('sample too long')
                return self[index + 1]
            try:
                img,bmask = self.keepratio_resize(img)
            except:
                print('Size error for %d' % index)
                return self[index + 1]
            img = img[:,:,np.newaxis]
            if self.transform:
                img = self.transform(img)
                bmask=self.transform(bmask)
            sample = {'image': img, 'label': label,"bmask":bmask}
            return sample
class lmdbDataset_repeatH(lmdbDataset):
    def __len__(self):
        return self.nSamples*self.repeat;

    def __getitem__(self, index):
        index%=self.nSamples;
        fromwhich = self.__fromwhich__()
        if self.global_state == 'Train':
            index = random.randint(0,self.maxlen - 1)
        index = index % self.lengths[fromwhich]
        assert index <= len(self), 'index range error'
        index += 1
        with self.envs[fromwhich].begin(write=False) as txn:
            img_key = 'image-%09d' % index
            try:
                imgbuf = txn.get(img_key.encode())
                buf = six.BytesIO()
                buf.write(imgbuf)
                buf.seek(0)
                img = Image.open(buf)
            except:
                print('Corrupted image for %d' % index)
                print("reponame: " ,self.roots[fromwhich]);
                return self[index + 1]

            label_key = 'label-%09d' % index
            try:
                label = str(txn.get(label_key.encode()).decode('utf-8'));
            except:
                print("reponame: " ,self.roots[fromwhich]);
            if len(label) > self.maxT-1 and self.global_state == 'Train':
                print('sample too long')
                return self[index + 1]
            # print(img)

            if(self.novert and img.size[0] / float(img.size[1])<1 ):
                # print(img.size)
                # print("vertical image");
                return self[index + 1]
            try:
                img,bmask = self.keepratio_resize(img)
            except:
                print('Size error for %d' % index)
                return self[index + 1]
            img = img[:,:,np.newaxis]
            if self.transform:
                img = self.transform(img)
                bmask = self.transform(bmask)

            sample = {'image': img, 'label': label,"bmask":bmask}
            return sample

class lmdbDataset_repeatS(lmdbDataset):
    def __len__(self):
        if(self.repeat==-1):
            return 2147483647;
        else:
            return self.nSamples*self.repeat;
    def init_etc(this):
        this.ccache={};
        this.occr_cnt={};
    def grab(self,fromwhich,index):
        with self.envs[fromwhich].begin(write=False) as txn:
            img_key = 'image-%09d' % index
            try:
                imgbuf = txn.get(img_key.encode())
                buf = six.BytesIO()
                buf.write(imgbuf)
                buf.seek(0)
                img = Image.open(buf)
            except:
                print('Corrupted image for %d' % index)
                print("reponame: " ,self.roots[fromwhich]);
                return self[index + 1]

            label_key = 'label-%09d' % index
            try:
                label = str(txn.get(label_key.encode()).decode('utf-8'));
            except:
                print("reponame: " ,self.roots[fromwhich]);
            if len(label) > self.maxT-1 and self.global_state == 'Train':
                print('sample too long')
                return self[index + 1]
            try:
                img,mask = self.keepratio_resize(img)
            except:
                print('Size error for %d' % index)
                return self[index + 1]
            img = img[:,:,np.newaxis]
            if self.transform:
                img = self.transform(img)
                mask=self.transform(mask)
            sample = {'image': img, 'label': label,"bmask":mask}

            return sample;
    def __getitem__(self, index):
        index%=self.nSamples;
        fromwhich = self.__fromwhich__()
        if self.global_state == 'Train':
            index = random.randint(0,self.maxlen - 1)
        index = index % self.lengths[fromwhich]
        assert index <= len(self), 'index range error'
        index += 1
        # prevent the common characters smacking the space.
        if len(self.ccache) and random.randint(0,10)>7:
            ks=list(self.ccache.keys());
            fs=[1./self.occr_cnt[k] for k in ks]
            k=random.choices(ks,fs)[0];
            # print(k,self.occr_cnt[k]);
            ridx,fwhich=self.ccache[k];
            sample=self.grab(ridx,fwhich);
        else:
            sample = self.grab(fromwhich, index);
            minoccr=np.inf;
            minK=None;
            for l in sample["label"]:
                if l not in self.occr_cnt:
                    self.occr_cnt[l]=0;
                self.occr_cnt[l] +=1;
                if(self.occr_cnt[l]<minoccr):
                    minoccr=self.occr_cnt[l];
                    minK=l
            if(minK):
                self.ccache[minK] = (fromwhich, index);

        for l in sample["label"]:
            self.occr_cnt[l]+=1;
        return sample

class colored_lmdbDataset_repeatS(lmdbDataset_repeatS):
    def grab(self,fromwhich,index):
        with self.envs[fromwhich].begin(write=False) as txn:
            img_key = 'image-%09d' % index
            try:
                imgbuf = txn.get(img_key.encode())
                buf = six.BytesIO()
                buf.write(imgbuf)
                buf.seek(0)
                img = Image.open(buf)
            except:
                print('Corrupted image for %d' % index)
                print("reponame: " ,self.roots[fromwhich]);
                return self[index + 1]

            label_key = 'label-%09d' % index
            try:
                label = str(txn.get(label_key.encode()).decode('utf-8'));
            except:
                print("reponame: " ,self.roots[fromwhich]);
            if len(label) > self.maxT-1 and self.global_state == 'Train':
                print('sample too long')
                return self[index + 1]
            # print(img)

            try:
                img,bmask = self.keepratio_resize(img)
            except:
                print('Size error for %d' % index)
                return self[index + 1]
            # img = img[:,:,np.newaxis]
            if self.transform:
                img = self.transform(img)
                bmask = self.transform(bmask)

            sample = {'image': img, 'label': label,"bmask":bmask}

            return sample;

    def keepratio_resize(self, img):
        img, bmask = neko_DAN_padding(img, None,
                                          img_width=self.img_width,
                                          img_height=self.img_height,
                                          target_ratio=self.target_ratio,
                                          qhb_aug=self.qhb_aug,gray=False)
        return img,bmask

class lmdbDataset_repeatHS(lmdbDataset_repeatS):
    def __len__(self):
        if(self.repeat<0):
            return 2147483647;
        return self.nSamples*self.repeat;
    def init_etc(this):
        this.ccache={};
        this.occr_cnt={};
    def grab(self,fromwhich,index):
        with self.envs[fromwhich].begin(write=False) as txn:
            img_key = 'image-%09d' % index
            try:
                imgbuf = txn.get(img_key.encode())
                buf = six.BytesIO()
                buf.write(imgbuf)
                buf.seek(0)
                img = Image.open(buf)
            except:
                print('Corrupted image for %d' % index)
                print("reponame: " ,self.roots[fromwhich]);
                return self[index + 1]

            label_key = 'label-%09d' % index
            try:
                label = str(txn.get(label_key.encode()).decode('utf-8'));
            except:
                print("reponame: " ,self.roots[fromwhich]);
            if len(label) > self.maxT-1 and self.global_state == 'Train':
                print('sample too long')
                return self[index + 1]
            # print(img)

            if(self.novert and img.size[0] / float(img.size[1])<1 ):
                # print(img.size)
                # print("vertical image");
                return self[index + 1]
            try:
                img,mask = self.keepratio_resize(img)
            except:
                print('Size error for %d' % index)
                return self[index + 1]

            if(len(img.shape)==2):
                img = img[:,:,np.newaxis]
            if self.transform:
                img= self.transform(img)
                mask=self.transform(mask)
            sample = {'image': img, 'label': label,"bmask":mask}

            return sample;

class colored_lmdbDataset_repeatHS(lmdbDataset_repeatHS):
    def keepratio_resize(self, img):
        img, bmask = neko_DAN_padding(img, None,
                                          img_width=self.img_width,
                                          img_height=self.img_height,
                                          target_ratio=self.target_ratio,
                                          qhb_aug=self.qhb_aug,gray=False)
        return img,bmask
    def grab(self,fromwhich,index):
        with self.envs[fromwhich].begin(write=False) as txn:
            img_key = 'image-%09d' % index
            try:
                imgbuf = txn.get(img_key.encode())
                buf = six.BytesIO()
                buf.write(imgbuf)
                buf.seek(0)
                img = Image.open(buf)
            except:
                print('Corrupted image for %d' % index)
                print("reponame: " ,self.roots[fromwhich]);
                return self[index + 1]

            label_key = 'label-%09d' % index
            try:
                label = str(txn.get(label_key.encode()).decode('utf-8'));
            except:
                print("reponame: " ,self.roots[fromwhich]);
            if len(label) > self.maxT-1 and self.global_state == 'Train':
                print('sample too long')
                return self[index + 1]
            # print(img)

            if(self.novert and img.size[0] / float(img.size[1])<1 ):
                # print(img.size)
                # print("vertical image");
                return self[index + 1]
            try:
                img,bmask = self.keepratio_resize(img)
            except:
                print('Size error for %d' % index)
                return self[index + 1]
            # img = img[:,:,np.newaxis]
            if self.transform:
                img = self.transform(img);
                bmask=self.transform(bmask);
            sample = {'image': img, 'label': label,"bmask":bmask}

            return sample;

#
# class lmdbDatasetTransform(lmdbDataset):
#     def __init__(self, roots=None, ratio=None, img_height = 32, img_width = 128,
#         transform=None, global_state='Test',maxT=25,repeat=1):
#         self.envs = []
#         self.roots=[];
#         self.maxT=maxT;
#         self.nSamples = 0
#         self.lengths = []
#         self.ratio = []
#         self.global_state = global_state
#         self.repeat=repeat;
#         self.totensor=torchvision.transforms.ToTensor();
#         for i in range(0,len(roots)):
#             env = lmdb.open(
#                 roots[i],
#                 max_readers=1,
#                 readonly=True,
#                 lock=False,
#                 readahead=False,
#                 meminit=False)
#             if not env:
#                 print('cannot creat lmdb from %s' % (roots[i]))
#                 sys.exit(0)
#             with env.begin(write=False) as txn:
#                 nSamples = int(txn.get('num-samples'.encode()))
#                 self.nSamples += nSamples
#             self.lengths.append(nSamples)
#             self.roots.append(roots[i]);
#             self.envs.append(env)
#
#         if ratio != None:
#             assert len(roots) == len(ratio) ,'length of ratio must equal to length of roots!'
#             for i in range(0,len(roots)):
#                 self.ratio.append(ratio[i] / float(sum(ratio)))
#         else:
#             for i in range(0,len(roots)):
#                 self.ratio.append(self.lengths[i] / float(self.nSamples))
#
#         self.transform = transform
#         self.maxlen = max(self.lengths)
#         self.img_height = img_height
#         self.img_width = img_width
#         # Issue12
#         self.target_ratio = img_width / float(img_height)
#
#
#
#     def __getitem__(self, index):
#         fromwhich = self.__fromwhich__()
#         if self.global_state == 'Train':
#             index = random.randint(0,self.maxlen - 1);
#
#         index = index % self.lengths[fromwhich]
#         assert index <= len(self), 'index range error'
#         index += 1
#         with self.envs[fromwhich].begin(write=False) as txn:
#             img_key = 'image-%09d' % index
#             try:
#                 imgbuf = txn.get(img_key.encode())
#                 buf = six.BytesIO()
#                 buf.write(imgbuf)
#                 buf.seek(0)
#                 img = Image.open(buf)
#                 img=self.transform(img)
#             except:
#                 print('Corrupted image for %d' % index)
#                 return self[index + 1]
#             label_key = 'label-%09d' % index
#             label = str(txn.get(label_key.encode()).decode('utf-8'));
#             if len(label) > self.maxT-1 and self.global_state == 'Train':
#                 print('sample too long')
#                 return self[index + 1]
#             try:
#                 img,bmask = self.keepratio_resize(img)
#             except:
#                 print('Size error for %d' % index)
#                 return self[index + 1]
#             img = img[:,:,np.newaxis]
#             if self.transform:
#                 img = self.totensor(img)
#                 bmask=self.totensor(bmask)
#             sample = {'image': img, 'label': label,"bmask":bmask}
#             return sample
