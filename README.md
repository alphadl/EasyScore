# score_with_bert
scoring a sentence with pretrained BERT model

you can download the official released chinese BERT model [chinese_L-12_H-768_A-12](https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip).

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
```

scoring senteces with NLL and PPL:
```
Initializing model...
#########The 1 th sent:###########
['2016全国高考卷答题模板'] NLL: 2.1058068908634193 PPL: 4.304384305590411
#########The 2 th sent:###########
['2016全国大考卷答题模板'] NLL: 2.795245075424673 PPL: 6.941488570912706
#########The 3 th sent:###########
['2016全国低考卷答题模板'] NLL: 3.1383238589224005 PPL: 8.805005217385805
```

probing token level predition and score with ``--prob_token``:
```
The 1 th position: {pred_token} ['['] 0.8748                     {golden_token} ['['] 0.8748
The 2 th position: {pred_token} ["'"] 0.9909                     {golden_token} ["'"] 0.9909
The 3 th position: {pred_token} ['2013'] 0.1611                  {golden_token} ['2016'] 0.1448
The 4 th position: {pred_token} ['全'] 0.6356                    {golden_token} ['全'] 0.6356
The 5 th position: {pred_token} ['国'] 0.9898                    {golden_token} ['国'] 0.9898
The 6 th position: {pred_token} ['中'] 0.4359                    {golden_token} ['高'] 0.3728
The 7 th position: {pred_token} ['考'] 0.9643                    {golden_token} ['考'] 0.9643
The 8 th position: {pred_token} ['试'] 0.1852                    {golden_token} ['卷'] 0.0009
The 9 th position: {pred_token} ['试'] 0.5127                    {golden_token} ['答'] 0.0823
The 10 th position: {pred_token} ['案'] 0.7375                   {golden_token} ['题'] 0.1855
The 11 th position: {pred_token} ['模'] 0.9187                   {golden_token} ['模'] 0.9187
The 12 th position: {pred_token} ['板'] 0.5225                   {golden_token} ['板'] 0.5225
The 13 th position: {pred_token} ["'"] 0.9911                    {golden_token} ["'"] 0.9911
The 14 th position: {pred_token} ['。'] 0.8935                   {golden_token} [']'] 0.0068
The 1 th position: {pred_token} ['['] 0.8748                     {golden_token} ['['] 0.8748
The 2 th position: {pred_token} ["'"] 0.9909                     {golden_token} ["'"] 0.9909
The 3 th position: {pred_token} ['2013'] 0.1611                  {golden_token} ['2016'] 0.1448
The 4 th position: {pred_token} ['全'] 0.6356                    {golden_token} ['全'] 0.6356
The 5 th position: {pred_token} ['国'] 0.9898                    {golden_token} ['国'] 0.9898
The 6 th position: {pred_token} ['中'] 0.4359                    {golden_token} ['高'] 0.3728
The 7 th position: {pred_token} ['考'] 0.9643                    {golden_token} ['考'] 0.9643
The 8 th position: {pred_token} ['试'] 0.1852                    {golden_token} ['卷'] 0.0009
The 9 th position: {pred_token} ['试'] 0.5127                    {golden_token} ['答'] 0.0823
The 10 th position: {pred_token} ['案'] 0.7375                   {golden_token} ['题'] 0.1855
The 11 th position: {pred_token} ['模'] 0.9187                   {golden_token} ['模'] 0.9187
The 12 th position: {pred_token} ['板'] 0.5225                   {golden_token} ['板'] 0.5225
The 13 th position: {pred_token} ["'"] 0.9911                    {golden_token} ["'"] 0.9911
The 14 th position: {pred_token} ['。'] 0.8935                   {golden_token} [']'] 0.0068

['2016全国高考卷答题模板'] NLL: 2.1058068908634193 PPL: 4.304384305590411
```
