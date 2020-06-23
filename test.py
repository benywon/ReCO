# -*- coding: utf-8 -*-
"""
 @Time    : 2020/6/23 下午1:43
 @FileName: test.py
 @author: 王炳宁
 @contact: wangbingning@sogou-inc.com
"""
import argparse

import torch

from model import Bert4ReCO
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument("--model_type", type=str, default='bert-base-chinese')
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument(
    "--fp16",
    action="store_true",
    default=True,
)
args = parser.parse_args()
model_type = args.model_type
batch_size = args.batch_size
test_data = load_file('data/test.{}.obj'.format(model_type.replace('/', '.')))
test_data = sorted(test_data, key=lambda x: len(x[0]))
model = Bert4ReCO(model_type)
model.load_state_dict(torch.load('checkpoint.{}.th'.format(model_type.replace('/', '.')), map_location='cpu'))
model.cuda()
if args.fp16:
    try:
        from apex import amp
    except ImportError:
        raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

    [model] = amp.initialize([model], opt_level='O1', verbosity=0)
model.eval()
total = len(test_data)
right = 0
with torch.no_grad():
    for i in tqdm(range(0, total, batch_size)):
        seq = [x[0] for x in test_data[i:i + batch_size]]
        labels = [x[1] for x in test_data[i:i + batch_size]]
        seq = padding(seq, pads=0, max_len=512)
        seq = torch.LongTensor(seq).cuda()
        predictions = model([seq, None])
        predictions = predictions.cpu()
        right += predictions.eq(torch.LongTensor(labels)).sum().item()
acc = 100 * right / total
print('test acc is {}'.format(acc))
