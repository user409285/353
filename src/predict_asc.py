import os
import logging
import argparse
import random
import json

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertForSequenceClassification
from pytorch_pretrained_bert.optimization import BertAdam

import absa_data_utils as data_utils
from absa_data_utils import ABSATokenizer
import modelconfig

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

class Predict_asc:
    def __init__(self, bert_model, model_pt, max_seq_length=100, eval_batch_size=8):
        self.tokenizer = BertTokenizer.from_pretrained(bert_model)
        self.processor = data_utils.AscProcessor()
        self.label_list = self.processor.get_labels()
        self.max_seq_length = max_seq_length
        self.eval_batch_size = eval_batch_size
        self.model = torch.load(model_pt)
        self.model.cuda()
        self.model.eval()

    def get_examples(self, lines):
        new_lines = {}
        for line in lines:
            new_lines[line['id']] = line
        return self.processor._create_examples(new_lines, 'test')

    def predict(self, lines):
        # lines: list,每一段话分解后的
        eval_examples = self.get_examples(lines)
        eval_features = data_utils.convert_examples_to_features(eval_examples, self.label_list, self.max_seq_length, self.tokenizer, "asc")
        #logger.info("***** Running evaluation *****")
        #logger.info("  Num examples = %d", len(eval_examples))
        #logger.info("  Batch size = %d", self.eval_batch_size)
        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
        eval_data = TensorDataset(all_input_ids, all_segment_ids, all_input_mask, all_label_ids)
        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=self.eval_batch_size)
        full_logits=[]
        full_label_ids=[]
        for step, batch in enumerate(eval_dataloader):
            batch = tuple(t.cuda() for t in batch)
            input_ids, segment_ids, input_mask, label_ids = batch
            
            with torch.no_grad():
                logits = self.model(input_ids, segment_ids, input_mask)

            logits = logits.detach().cpu().numpy()
            label_ids = [np.argmax(lg) for lg in logits.tolist()]
            #label_ids = label_ids.cpu().numpy()

            full_logits.extend(logits.tolist())
            full_label_ids.extend(label_ids)
            #full_label_ids.append(label_id)
        return {'logits': full_logits, 'label_ids': full_label_ids}
