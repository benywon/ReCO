# -*- coding: utf-8 -*-
"""
 @Time    : 2020/6/23 上午10:15
 @FileName: prepare_data.py
 @author: 王炳宁
 @contact: wangbingning@sogou-inc.com
"""
import json
import random

from transformers import BertTokenizer

from utils import *

tokenizer = None


def get_shuffled_answer(alternatives):
    answers_index = [0, 1, 2]
    random.shuffle(answers_index)
    alternatives = [alternatives[x] for x in answers_index]
    label = list(answers_index).index(0)
    return alternatives, label


def get_one_sample_features(one):
    alternatives, label = get_shuffled_answer(one['alternatives'].split('|'))
    query = one['query']
    paragraph = clean(one['passage'])
    alt_ids = [y for x in alternatives for y in [1] + tokenizer.encode(x)]
    seq_ids = alt_ids + [2] + tokenizer.encode(query) + [
        tokenizer.sep_token_id]
    seq_ids += tokenizer.encode(paragraph, max_length=tokenizer.max_len - len(seq_ids))
    return [seq_ids, label]


def convert_to_features(filename):
    with open(filename, encoding='utf-8') as f:
        raw = json.load(f)
        data = multi_process(get_one_sample_features, raw)
    print('get {} with {} samples'.format(filename, len(data)))
    return data


def prepare_bert_data(model_type='bert-base-chinese'):
    global tokenizer
    tokenizer = BertTokenizer.from_pretrained(model_type)
    if not os.path.exists('data/test.{}.obj'.format(model_type.replace('/', '.'))):
        test_data = convert_to_features('data/ReCO/ReCO.testa.json')
        dump_file(test_data, 'data/test.{}.obj'.format(model_type.replace('/', '.')))
    if not os.path.exists('data/valid.{}.obj'.format(model_type.replace('/', '.'))):
        valid_data = convert_to_features('data/ReCO/ReCO.validationset.json')
        dump_file(valid_data, 'data/valid.{}.obj'.format(model_type.replace('/', '.')))
    if not os.path.exists('data/train.{}.obj'.format(model_type.replace('/', '.'))):
        train_data = convert_to_features('data/ReCO/ReCO.trainingset.json')
        dump_file(train_data, 'data/train.{}.obj'.format(model_type.replace('/', '.')))
