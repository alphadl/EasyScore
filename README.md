# score_with_bert
scoring a sentence with pretrained BERT model

**usage:**

```
python score_with_pretrained_BERT.py -h

  --bert_path BERT_PATH
                        pre-downloaded bert path. for example, we download the
                        chinese pretrained BERT model with path name
                        'chinese_L-12_H-768_A-12'
  --gpu_id GPU_ID       gpu_id. The default setting is using the first gpu
  --dev_data DEV_DATA   the corpus file you want to calculate.
  --prob_token          print the predicted word and its probability and this
                        position's corresponding gold token
  --show_model          logging the pretrained BERT model
'''
