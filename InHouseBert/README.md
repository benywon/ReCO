## In-house  BERT Implementations

This is our in-house inplementation of BERT which is optimized for Chinese language representations.

The model is trained on 4.7 billion Chinese web pages (nearly 1.3TB data).

 However,  for the license limitations, we can only release the BERT_base model.



## Requirements

#### -Apex

You should install apex with optimized multi-head attention (which is 1.6 times faster than official Multi-head attention)

https://github.com/NVIDIA/apex/tree/master/apex/contrib/multihead_attn

`pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" --global-option="--fast_multihead_attn" ./`

#### -Pre-trained models

Download the pre-trained BERT model from here:

https://drive.google.com/file/d/1eqC_TQQjEhvOtAP5wh5k023mr8AchWVi/view?usp=sharing



## Train and Inference

1) pre-process data

`python3 prepare_data.py`



2) Training

`python3 -m torch.distributed.launch --nproc_per_node=8 train.py`



3) Inference 

`python3 test.py`









