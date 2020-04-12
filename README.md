# score_with_bert

Scoring a sentence with pretrained BERT model, setting your pretrained BERT model path with ``--bert_path``, and make sure the path of the sentence to be scored is provided with ``--dev_data``.

For Chinese sentences scoring, you can download the official released chinese BERT model [chinese_L-12_H-768_A-12](https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip).

As for English sentences, you can download all 24 from [here][all], or individually from the table below:

|   |H=128|H=256|H=512|H=768|
|---|:---:|:---:|:---:|:---:|
| **L=2**  |[**2/128 (BERT-Tiny)**][2_128]|[2/256][2_256]|[2/512][2_512]|[2/768][2_768]|
| **L=4**  |[4/128][4_128]|[**4/256 (BERT-Mini)**][4_256]|[**4/512 (BERT-Small)**][4_512]|[4/768][4_768]|
| **L=6**  |[6/128][6_128]|[6/256][6_256]|[6/512][6_512]|[6/768][6_768]|
| **L=8**  |[8/128][8_128]|[8/256][8_256]|[**8/512 (BERT-Medium)**][8_512]|[8/768][8_768]|
| **L=10** |[10/128][10_128]|[10/256][10_256]|[10/512][10_512]|[10/768][10_768]|
| **L=12** |[12/128][12_128]|[12/256][12_256]|[12/512][12_512]|[**12/768 (BERT-Base)**][12_768]|


[2_128]: https://storage.googleapis.com/bert_models/2020_02_20/uncased_L-2_H-128_A-2.zip
[2_256]: https://storage.googleapis.com/bert_models/2020_02_20/uncased_L-2_H-256_A-4.zip
[2_512]: https://storage.googleapis.com/bert_models/2020_02_20/uncased_L-2_H-512_A-8.zip
[2_768]: https://storage.googleapis.com/bert_models/2020_02_20/uncased_L-2_H-768_A-12.zip
[4_128]: https://storage.googleapis.com/bert_models/2020_02_20/uncased_L-4_H-128_A-2.zip
[4_256]: https://storage.googleapis.com/bert_models/2020_02_20/uncased_L-4_H-256_A-4.zip
[4_512]: https://storage.googleapis.com/bert_models/2020_02_20/uncased_L-4_H-512_A-8.zip
[4_768]: https://storage.googleapis.com/bert_models/2020_02_20/uncased_L-4_H-768_A-12.zip
[6_128]: https://storage.googleapis.com/bert_models/2020_02_20/uncased_L-6_H-128_A-2.zip
[6_256]: https://storage.googleapis.com/bert_models/2020_02_20/uncased_L-6_H-256_A-4.zip
[6_512]: https://storage.googleapis.com/bert_models/2020_02_20/uncased_L-6_H-512_A-8.zip
[6_768]: https://storage.googleapis.com/bert_models/2020_02_20/uncased_L-6_H-768_A-12.zip
[8_128]: https://storage.googleapis.com/bert_models/2020_02_20/uncased_L-8_H-128_A-2.zip
[8_256]: https://storage.googleapis.com/bert_models/2020_02_20/uncased_L-8_H-256_A-4.zip
[8_512]: https://storage.googleapis.com/bert_models/2020_02_20/uncased_L-8_H-512_A-8.zip
[8_768]: https://storage.googleapis.com/bert_models/2020_02_20/uncased_L-8_H-768_A-12.zip
[10_128]: https://storage.googleapis.com/bert_models/2020_02_20/uncased_L-10_H-128_A-2.zip
[10_256]: https://storage.googleapis.com/bert_models/2020_02_20/uncased_L-10_H-256_A-4.zip
[10_512]: https://storage.googleapis.com/bert_models/2020_02_20/uncased_L-10_H-512_A-8.zip
[10_768]: https://storage.googleapis.com/bert_models/2020_02_20/uncased_L-10_H-768_A-12.zip
[12_128]: https://storage.googleapis.com/bert_models/2020_02_20/uncased_L-12_H-128_A-2.zip
[12_256]: https://storage.googleapis.com/bert_models/2020_02_20/uncased_L-12_H-256_A-4.zip
[12_512]: https://storage.googleapis.com/bert_models/2020_02_20/uncased_L-12_H-512_A-8.zip
[12_768]: https://storage.googleapis.com/bert_models/2020_02_20/uncased_L-12_H-768_A-12.zip
[all]: https://storage.googleapis.com/bert_models/2020_02_20/all_bert_models.zip

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
