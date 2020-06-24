based on the IJCAI-2018 paper [Multiway Attention Networks for Modeling Sentence Pairs](https://www.ijcai.org/Proceedings/2018/0613.pdf) and BiDAF ([Bidirectional Attention Flow for Machine Comprehension](https://arxiv.org/abs/1611.01603)) 

## Reuqirements

python 2.7

pytorch 0.4.1  

jieba

## Preprocessing and Training

python train.py --cuda

You can either set the model to BiDAF or MwAN in the train.py

## Test

python inference.py --data data/ai_challenger_oqmrc_testa_20180816/ai_challenger_oqmrc_testa.json --output prediction.txt
