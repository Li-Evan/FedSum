import os
import bisect
import random

import torch
import glob
import numpy as np
from tool.logger import *
from torch.utils.data import Dataset
from data_preprocess.preprocess_dataset \
    import process_GovernmentReport, process_Wikihow, process_PubMed, process_Reddit


# import sys
# # 将指定文件夹添加到Python的搜索路径中
# sys.path.insert(0, r"E:\Lab\论文代码\open_source_version_code")

class BERTSUMDataset(Dataset):
    def _pad(self, data, pad_id, width=-1):
        if (width == -1):
            width = max(len(d) for d in data)
        rtn_data = [d + [pad_id] * (width - len(d)) for d in data]
        return rtn_data

    def __init__(self, attribute_dict, data_name):
        pre_src = attribute_dict["src"]
        pre_tgt = attribute_dict["tgt"]
        pre_segs = attribute_dict["segs"]
        pre_clss = attribute_dict["clss"]
        pre_src_sent_labels = attribute_dict["src_sent_labels"]

        self.src = torch.tensor(self._pad(pre_src, 0))
        self.tgt = torch.tensor(self._pad(pre_tgt, 0))
        self.segs = torch.tensor(self._pad(pre_segs, 0))
        self.mask_src = ~(self.src == 0)  # mask_src = 1 - (src == 0)
        self.mask_tgt = ~(self.tgt == 0)  # mask_tgt = 1 - (tgt == 0)

        self.clss = torch.tensor(self._pad(pre_clss, -1))
        self.src_sent_labels = torch.tensor(self._pad(pre_src_sent_labels, 0))
        self.mask_cls = ~(self.clss == -1)  # mask_cls = 1 - (clss == -1)
        self.clss[self.clss == -1] = 0

        self.src_txt = attribute_dict["src_txt"]
        self.tgt_txt = attribute_dict["tgt_txt"]
        self.tag = [data_name] * len(self.src)

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        return {
            "src": self.src[idx],
            "tgt": self.tgt[idx],
            "src_sent_labels": self.src_sent_labels[idx],
            "segs": self.segs[idx],
            "clss": self.clss[idx],
            "mask_src": self.mask_src[idx],
            "mask_tgt": self.mask_tgt[idx],
            "mask_cls": self.mask_cls[idx],
            "src_txt": self.src_txt[idx],
            "tgt_txt": self.tgt_txt[idx],
            "tag": self.tag[idx]
        }

    def reshape(self, max_len_dict):
        max_len_mask_cls = max_len_dict.get('mask_cls', None)
        max_len_clss = max_len_dict.get('clss', None)
        max_len_tgt = max_len_dict.get('tgt', None)

        temp_mask_cls = self.mask_cls.numpy().tolist()
        temp_src_sent_labels = self.src_sent_labels.numpy().tolist()
        temp_clss = self.clss.numpy().tolist()
        temp_tgt= self.tgt.numpy().tolist()

        self.mask_cls = torch.tensor(self._pad(temp_mask_cls, False, max_len_mask_cls))
        self.clss = torch.tensor(self._pad(temp_clss, 0, max_len_mask_cls))
        self.src_sent_labels = torch.tensor(self._pad(temp_src_sent_labels, 0, max_len_mask_cls))
        self.tgt= torch.tensor(self._pad(temp_tgt, 0, max_len_tgt))


def get_CNNDM_dataset(data_path, corpus_type, param_dict, only_one=False, source='CNNDM'):
    def preprocess(data_batch):
        max_tgt_len = 140
        max_pos = 512

        src_list = []
        tgt_list = []
        src_sent_labels_list = []
        segs_list = []
        clss_list = []
        src_txt_list = []
        tgt_txt_list = []
        for i in range(len(data_batch["src"])):
            src = data_batch["src"][i]
            tgt = data_batch["tgt"][i][:max_tgt_len][:-1] + [2]
            src_sent_labels = data_batch["src_sent_labels"][i]
            segs = data_batch["segs"][i]
            clss = data_batch["clss"][i]
            src_txt = data_batch["src_txt"][i]
            tgt_txt = data_batch["tgt_txt"][i]

            end_id = [src[-1]]
            src = src[:-1][:max_pos - 1] + end_id
            segs = segs[:max_pos]
            max_sent_id = bisect.bisect_left(clss, max_pos)
            src_sent_labels = src_sent_labels[:max_sent_id]
            clss = clss[:max_sent_id]
            # src_txt = src_txt[:max_sent_id]

            src_list.append(src)
            tgt_list.append(tgt)
            src_sent_labels_list.append(src_sent_labels)
            segs_list.append(segs)
            clss_list.append(clss)
            src_txt_list.append(src_txt)
            tgt_txt_list.append(tgt_txt)

        return {
            'src': src_list,
            'tgt': tgt_list,
            'src_sent_labels': src_sent_labels_list,
            "segs": segs_list,
            "clss": clss_list,
            "src_txt": src_txt_list,
            "tgt_txt": tgt_txt_list,
        }

    if corpus_type == "test":
        if param_dict["tt"] == "3":
            pts = sorted(glob.glob(data_path + '/' + corpus_type + '.[0]*.pt'))
        elif param_dict["tt"] == "2":
            pts = sorted(glob.glob(data_path + '/' + corpus_type + '.[0].bert.pt'))
        elif param_dict["tt"] == "few":
            pts = sorted(glob.glob(data_path + '/' + corpus_type + '.[0-9].bert.pt'))
        else:
            pts = sorted(glob.glob(data_path + '/' + corpus_type + '.[0-9].bert.pt'))

    else:
        if param_dict["tt"] == "0":
            pts = sorted(glob.glob(data_path + '/' + corpus_type + '.[0-9]*.pt'))
        elif param_dict["tt"] == "1":
            pts = sorted(glob.glob(data_path + '/' + corpus_type + '.[0-3].bert.pt'))
        elif param_dict["tt"] == "few":
            # ori_pts = sorted(glob.glob(data_path + '/' + corpus_type + '.[0-9].bert.pt'))  # 标准写法
            ori_pts = sorted(glob.glob(data_path + '/' + corpus_type + '.[0].bert.pt'))  # 为了控制实验变量使用
            pts = [random.choice(ori_pts)]
        else:
            pts = sorted(glob.glob(data_path + '/' + corpus_type + '.[0-9]*.pt'))

    full_dataset = []
    for idx, pt in enumerate(pts):
        pieces = torch.load(pt)
        # print(f"gpu consume:{torch.cuda.memory_allocated()}")
        logger.info('Loading %s dataset from %s, number of examples: %d' % (corpus_type, pt, len(pieces)))
        full_dataset += pieces


    attribute_dict = preprocess(
        {
            'src': [dic["src"] for dic in full_dataset],
            'tgt': [dic["tgt"] for dic in full_dataset],
            'src_sent_labels': [dic["src_sent_labels"] for dic in full_dataset],
            "segs": [dic["segs"] for dic in full_dataset],
            "clss": [dic["clss"] for dic in full_dataset],
            "src_txt": [dic["src_txt"] for dic in full_dataset],
            "tgt_txt": [dic["tgt_txt"] for dic in full_dataset],

        })

    B_Dataset = BERTSUMDataset(attribute_dict=attribute_dict, data_name=source)

    return B_Dataset


def get_WikiHow_dataset(data_path, corpus_type, param_dict, only_one=True, source='WikiHow'):
    if not os.path.exists(data_path):
        os.mkdir(data_path)

    if "val" in corpus_type:
        corpus_type = "valid"


    if len(glob.glob(data_path + '/' + corpus_type + '.[0-9]*.pt')) == 0:
        process_Wikihow()
    return get_CNNDM_dataset(data_path, corpus_type, param_dict, only_one, source)


def get_GovernmentReport_dataset(data_path, corpus_type, param_dict, only_one=True, source='GovernmentReport'):
    if not os.path.exists(data_path):
        os.mkdir(data_path)

    if "val" in corpus_type:
        corpus_type = "val"


    if len(glob.glob(data_path + '/' + corpus_type + '.[0-9]*.pt')) == 0:
        process_GovernmentReport()

    return get_CNNDM_dataset(data_path, corpus_type, param_dict, only_one, source)


def get_PubMed_dataset(data_path, corpus_type, param_dict, only_one=True, source='PubMed'):
    if not os.path.exists(data_path):
        os.mkdir(data_path)

    if "val" in corpus_type:
        corpus_type = "val"

    if len(glob.glob(data_path + '/' + corpus_type + '.[0-9]*.pt')) == 0:
        process_PubMed()

    return get_CNNDM_dataset(data_path, corpus_type, param_dict, only_one, source)

def get_Reddit_dataset(data_path, corpus_type, param_dict, only_one=True, source='Reddit'):
    if not os.path.exists(data_path):
        os.mkdir(data_path)

    if "val" in corpus_type:
        corpus_type = "val"

    if len(glob.glob(data_path + '/' + corpus_type + '.[0-9]*.pt')) == 0:
        process_Reddit()

    return get_CNNDM_dataset(data_path, corpus_type, param_dict, only_one, source)
def get_Mixtape_dataset(data_paths, corpus_type, param_dict, only_one=True, ):
    datasets = []
    for data_path in data_paths:
        if "PubMed" in data_path:
            datasets.append(get_PubMed_dataset(data_path, corpus_type, param_dict, only_one, "PubMed"))
        elif "GovernmentReport" in data_path:
            datasets.append(
                get_GovernmentReport_dataset(data_path, corpus_type, param_dict, only_one, "GovernmentReport"))
        elif "WikiHow" in data_path:
            datasets.append(get_WikiHow_dataset(data_path, corpus_type, param_dict, only_one, "WikiHow"))
        elif "CNNDM" in data_path:
            datasets.append(get_CNNDM_dataset(data_path, corpus_type, param_dict, only_one, "CNNDM"))

    max_len_dict = {
        "mask_cls": max(data.mask_cls.shape[1] for data in datasets),
        "clss": max(data.clss.shape[1] for data in datasets),
        "src_sent_labels": max(data.src_sent_labels.shape[1] for data in datasets),
        "tgt": max(data.tgt.shape[1] for data in datasets),
    }

    for data in datasets:
        data.reshape(max_len_dict)
    mix_dataset = torch.utils.data.ConcatDataset(datasets)

    return mix_dataset



if __name__ == '__main__':
    print("Testing")
    # data_path = '../dataset/DRUG/'
    # mask_s1_flag = False
    # mask_s2_flag = False
    # mask_s1_s2_flag = False
    # qq, bb = get_DRUG_dataset(data_path, mask_s1_flag, mask_s2_flag, mask_s1_s2_flag)

    # training_dataset, testing_dataset = get_COMPAS_dataset("../dataset/COMPAS")

    # aa = get_CNNDM_dataset("../dataset/GovernmentReport", corpus_type="train")
    aa = get_GovernmentReport_dataset("../dataset/test_gov", param_dict={}, corpus_type="train")
    print(aa)
    print(1)