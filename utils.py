# -*- coding: utf-8 -*-
"""
 @Time    : 2019/11/20 下午6:14
 @FileName: utils.py
 @author: 王炳宁
 @contact: wangbingning@sogou-inc.com
"""
import os
import pickle
import re
import numpy as np
import multiprocessing
from joblib import Parallel, delayed
from tqdm import tqdm


def multi_process(func, lst, num_cores=multiprocessing.cpu_count(), backend='multiprocessing'):
    workers = Parallel(n_jobs=num_cores, backend=backend)
    output = workers(delayed(func)(one) for one in tqdm(lst))
    return [x for x in output if x]


def DBC2SBC(ustring):
    rstring = ""
    for uchar in ustring:
        inside_code = ord(uchar)
        if inside_code == 0x3000:
            inside_code = 0x0020
        else:
            inside_code -= 0xfee0
        if not (0x0021 <= inside_code <= 0x7e):
            rstring += uchar
            continue
        rstring += chr(inside_code)
    return rstring


def clean(txt):
    txt = DBC2SBC(txt)
    txt = txt.lower()
    return re.sub('\s*', '', txt)


def dump_file(obj, filename):
    f = open(filename, 'wb')
    pickle.dump(obj, f)


def load_file(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data


def get_dir_files(dirname):
    L = []
    for root, dirs, files in os.walk(dirname):
        for file in files:
            L.append(os.path.join(root, file))
    return L


def padding(sequence, pads=0, max_len=None, dtype='int32'):
    v_length = [len(x) for x in sequence]  # every sequence length
    seq_max_len = max(v_length)
    if (max_len is None) or (max_len > seq_max_len):
        max_len = seq_max_len
    x = (np.ones((len(sequence), max_len)) * pads).astype(dtype)
    for idx, s in enumerate(sequence):
        trunc = s[:max_len]
        x[idx, :len(trunc)] = trunc
    return x
