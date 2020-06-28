# -*- coding: utf-8 -*-
"""
 @Time    : 2020/6/24 下午6:16
 @FileName: train.py
 @author: 王炳宁
 @contact: wangbingning@sogou-inc.com
"""

import argparse
import sys

sys.path.append("../..")
sys.path.append("..")
from tasks.ReCO.model import BERT
from utils import *
import torch.distributed as dist

torch.manual_seed(100)
np.random.seed(200)
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--epoch", type=int, default=10)
parser.add_argument("--lr", type=float, default=4.0e-5)
parser.add_argument("--max_grad_norm", type=float, default=0.2)
parser.add_argument("--model_type", type=str, default="bert-base-chinese-new")
parser.add_argument(
    "--fp16",
    action="store_true",
    default=True,
)
parser.add_argument("--local_rank", type=int, default=-1)
args = parser.parse_args()
model_type = args.model_type
local_rank = args.local_rank
if local_rank >= 0:
    torch.distributed.init_process_group(backend='nccl',
                                         init_method='env://')
    torch.cuda.set_device(args.local_rank)

data = load_file('data/train.{}.obj'.format(model_type.replace('/', '.')))
valid_data = load_file('data/valid.{}.obj'.format(model_type.replace('/', '.')))
valid_data = sorted(valid_data, key=lambda x: len(x[0]))
batch_size = args.batch_size
n_embedding = 128
n_hidden = 768
n_layer = 12
n_head = 12
vocab_size = 50000
model = BERT(vocab_size, n_embedding, n_hidden, n_layer, n_head)
state_dict = load_file('model.bert.base.th')
for name, para in model.named_parameters():
    if name not in state_dict:
        print('{} not load'.format(name))
        continue
    para.data = torch.FloatTensor(state_dict[name])
model.cuda()
optimizer = torch.optim.AdamW(model.parameters(),
                              weight_decay=0.01,
                              lr=args.lr)

if args.fp16:
    try:
        from apex import amp
    except ImportError:
        raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

    model, optimizer = amp.initialize(model, optimizer, opt_level='O2', verbosity=0)

if local_rank >= 0:
    try:
        import apex
    except ImportError:
        raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use parallel training.")
    model = apex.parallel.DistributedDataParallel(model)


def get_shuffle_data():
    pool = {}
    for one in data:
        length = len(one[0]) // 5
        if length not in pool:
            pool[length] = []
        pool[length].append(one)
    for one in pool:
        np.random.shuffle(pool[one])
    length_lst = list(pool.keys())
    np.random.shuffle(length_lst)
    whole_data = [x for y in length_lst for x in pool[y]]
    if local_rank >= 0:
        remove_data_size = len(whole_data) % dist.get_world_size()
        thread_data = [whole_data[x + args.local_rank] for x in
                       range(0, len(whole_data) - remove_data_size, dist.get_world_size())]
        return thread_data
    return whole_data


def iter_printer(total, epoch):
    if local_rank >= 0:
        if local_rank == 0:
            return tqdm(range(0, total, batch_size), desc='epoch {}'.format(epoch))
        else:
            return range(0, total, batch_size)
    else:
        return tqdm(range(0, total, batch_size), desc='epoch {}'.format(epoch))


def train(epoch):
    model.train()
    train_data = get_shuffle_data()
    total = len(train_data)
    for i in iter_printer(total, epoch):
        seq = [x[0] for x in train_data[i:i + batch_size]]
        label = [x[1] for x in train_data[i:i + batch_size]]
        seq, _ = padding(seq, pads=0, max_len=512)
        seq = torch.LongTensor(seq).cuda()
        label = torch.LongTensor(label).cuda()
        loss = model([seq, label])
        if args.fp16:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
        optimizer.zero_grad()


def evaluation(epoch):
    model.eval()
    total = len(valid_data)
    right = 0
    with torch.no_grad():
        for i in iter_printer(total, epoch):
            seq = [x[0] for x in valid_data[i:i + batch_size]]
            labels = [x[1] for x in valid_data[i:i + batch_size]]
            seq, _ = padding(seq, pads=0, max_len=512)
            seq = torch.LongTensor(seq).cuda()
            predictions = model([seq, None])
            predictions = predictions.cpu()
            right += predictions.eq(torch.LongTensor(labels)).sum().item()
    acc = 100 * right / total
    print('epoch {} eval acc is {}'.format(epoch, acc))
    return acc


best_acc = 0.0
for epo in range(args.epoch):
    train(epo)
    if local_rank == -1 or local_rank == 0:
        accuracy = evaluation(epo)
        if accuracy > best_acc:
            best_acc = accuracy
            with open('checkpoint.{}.th'.format(model_type.replace('/', '.')), 'wb') as f:
                state_dict = model.module.state_dict() if args.fp16 else model.state_dict()
                torch.save(state_dict, f)
