# ReCO
ReCO: A Large Scale Chinese Reading Comprehension Dataset on Opinion

##Data
Dataset is available at https://drive.google.com/drive/folders/1rOAoKcLhMhge9uVQFM2_D1EU0AjnpWFa?usp=sharing

download the data and put the json files to the `data/ReCO` directory
###Stats
| Train | Dev |  Test-a | Test-b |
| ------------- | ------------- |------------- |------------- |
| 250,000  | 30,000  | 10,000  |10,000  |

##Requirenments
transformers
torch>=1.3.0
tqdm
joblib
apex(for mixed-precision training)
## Train and Test
For BiDAF and other types of model, you can refers to our [ai-challenger](https://github.com/AIChallenger/AI_Challenger_2018/tree/master/Baselines/opinion_questions_machine_reading_comprehension2018_baseline) public code for more details.
This code only contains the training script for Bert.  
###Train
For single node training:  
python3 train.py --model_type=bert-base-chinese  
for multiple nodes distributed training:  
python3 -m torch.distributed.launch --nproc_per_node=8 train.py --model_type=bert-base-chinese  

model_type could be either `bert-base-chinese` (for Bert_base) or `clue/roberta_chinese_large` (for Bert_large)
If you want to use the original doc as the context, you can set the `clean(one['passage'])` in prepare_data.py line 29 to `clean(one['doc'])`.
 
###Test
python3 test.py --model_type=bert-base-chinese

##Results
Doc level  

| Model | Dev |  Test-a |
| ------------- | ------------- |------------- |
| [BiDAF](https://github.com/AIChallenger/AI_Challenger_2018/tree/master/Baselines/opinion_questions_machine_reading_comprehension2018_baseline)  | 55.8  | 56.4  |
| [Bert-Base](https://huggingface.co/bert-base-chinese)  | 61.4  | 61.1  |
| [Bert-Large](https://huggingface.co/bert-base-chinese)  | 65.7  | 65.3  |
| Human  | --  | 88.0  |

Evidence level  

| Model | Dev |  Test-a |
| ------------- | ------------- |------------- |
| [BiDAF](https://github.com/AIChallenger/AI_Challenger_2018/tree/master/Baselines/opinion_questions_machine_reading_comprehension2018_baseline)  | 68.9  | 68.4  |
| [Bert-Base](https://huggingface.co/bert-base-chinese)  | 76.3  | 77.1  |
| [Bert-Large](https://huggingface.co/bert-base-chinese)  | 78.7  | 79.2  |
| [ALBert-tiny](https://huggingface.co/voidful/albert_chinese_tiny)  | 70.9  | 70.4  |
| [ALBert-base](https://huggingface.co/voidful/albert_chinese_base)  | 76.9  | 77.3  |
| [ALBert-xxLarge](https://huggingface.co/voidful/albert_chinese_xxlarge)  | 80.8  | 81.2  |
| Human  | --  | 91.5  |


## Citation
If you use ReCO in your research, please cite our work with the following BibTex Entry
```
@inproceedings{DBLP:conf/aaai/WangYZXW20,
  author    = {Bingning Wang and
               Ting Yao and
               Qi Zhang and
               Jingfang Xu and
               Xiaochuan Wang},
  title     = {ReCO: {A} Large Scale Chinese Reading Comprehension Dataset on Opinion},
  booktitle = {The Thirty-Fourth {AAAI} Conference on Artificial Intelligence, {AAAI}
               2020, The Thirty-Second Innovative Applications of Artificial Intelligence
               Conference, {IAAI} 2020, The Tenth {AAAI} Symposium on Educational
               Advances in Artificial Intelligence, {EAAI} 2020, New York, NY, USA,
               February 7-12, 2020},
  pages     = {9146--9153},
  publisher = {{AAAI} Press},
  year      = {2020},
  url       = {https://aaai.org/ojs/index.php/AAAI/article/view/6450},
  timestamp = {Thu, 04 Jun 2020 13:18:48 +0200},
  biburl    = {https://dblp.org/rec/conf/aaai/WangYZXW20.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```
