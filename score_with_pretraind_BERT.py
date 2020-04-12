import os
import argparse

import numpy as np
import pandas as pd

import torch
from transformers import BertTokenizer, BertForPreTraining, BertConfig
import logging

def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bert_path", type=str, default="chinese_L-12_H-768_A-12", \
        help="pre-downloaded bert path. for example, we download the chinese pretrained BERT model with path name 'chinese_L-12_H-768_A-12'")
    parser.add_argument("--gpu_id", type=int, default=0, \
        help="gpu_id. The default setting is using the first gpu")
    parser.add_argument("--dev_data", type=str, default="/***/test.zh.tsv", \
        help="the corpus file you want to calculate.")
    parser.add_argument("--prob_token", action='store_true', default=False, \
        help="print the predicted word and its probability and this position's corresponding gold token")
    parser.add_argument("--show_model", action='store_true', default=False, \
        help="logging the pretrained BERT model")
    return parser.parse_args()

class Scoring(object):
    def __init__(self, BERT_PATH):
        self.config = BertConfig.from_json_file(BERT_PATH+"/bert_config.json")
        self.model = BertForPreTraining.from_pretrained(BERT_PATH+"/bert_model.ckpt", from_tf=True, config=self.config)
        self.tokenizer = BertTokenizer(BERT_PATH+"/vocab.txt")
        self.model.eval()
        self.model.cuda(args.gpu_id)

    def sentence_preprocese(self, text):
        tokenized_text = np.array(self.tokenizer.tokenize(text))
        find_sep = np.argwhere(tokenized_text == '[SEP]')
        segments_ids = np.zeros(tokenized_text.shape, dtype=int)
        if find_sep.size == 1:
            start_point = 1
        else:
            start_point = find_sep[0, 0] + 1
            segments_ids[start_point:] = 1

        end_point = tokenized_text.size - 1

        tokenized_text = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        masked_texts = []

        # mask with l2r fashion
        for masked_index in range(start_point, end_point):
            new_tokenized_text = np.array(tokenized_text, dtype=int)
            new_tokenized_text[masked_index] = self.tokenizer.convert_tokens_to_ids(['[MASK]'])[0]
            masked_texts.append(new_tokenized_text)
        
        # copy the segments_ids
        segments_ids = np.tile(segments_ids, (end_point - start_point, 1))

        return masked_texts, segments_ids, start_point, end_point, tokenized_text[start_point:end_point]

    def metric(self, text):
        indexed_tokens, segments_ids, start_point, end_point, real_indexs = self.sentence_preprocese(text)

        tokens_tensor = torch.tensor(indexed_tokens)
        segments_tensors = torch.tensor(segments_ids)

        tokens_tensor = tokens_tensor.cuda(args.gpu_id)
        segments_tensors = segments_tensors.cuda(args.gpu_id)
        
        # model return: tuple()
        # 1. prediction_scores (batch_size X sequence_length X config.vocab_size); 
        # 2. seq_relationship_scores (batch_size X 2)

        with torch.no_grad():
            outputs = self.model(tokens_tensor, token_type_ids=segments_tensors)
            predictions = torch.softmax(outputs[0], -1)

        
        log_likelihood = 0
        
        # cumulated negative log likelihood

        for i, step in enumerate(range(start_point, end_point)):
            predicted_index = torch.argmax(predictions[i, step]).item()
            predicted_token = self.tokenizer.convert_ids_to_tokens([predicted_index])

            real_pos_prob = predictions[i, step, real_indexs[i]].item()
            real_token = self.tokenizer.convert_ids_to_tokens([real_indexs[i]])
            
            if args.prob_token == True:
            	print("The",i+1,"th position: {pred_token}",predicted_token, round(predictions[i, step, predicted_index].item(), 4), \
                	"\t\t\t {golden_token}", real_token, round(real_pos_prob, 4))

            log_likelihood += np.log2(real_pos_prob)
        
        prob = np.exp2(log_likelihood)
        nll = - log_likelihood / (end_point - start_point)
        ppl = np.exp2(nll)

        return nll, ppl


if __name__ == '__main__':
    args = parse_config()

    # model construction
    print("Initializing model...")
    score = Scoring(args.bert_path)

    # data formatter
    text_formatter = lambda x: "[CLS] {} [SEP]".format(x)

    # data preparation
    packed_data = pd.read_csv(args.dev_data, header=None, sep='\t').values
    for idx, t in enumerate(packed_data):
        print("#########The",idx+1,"th sent:###########")
        print(t, "NLL:",score.metric(text_formatter(t))[0], "PPL:",score.metric(text_formatter(t))[1])
