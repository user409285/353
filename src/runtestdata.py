import json
from tqdm import tqdm

import pandas as pd
import predict_asc

bert_model = '/home/xianjun/github/BERT-for-RRC-ABSA/pt_model/finance_pt/'
model_pt = '/home/xianjun/github/BERT-for-RRC-ABSA/run/pt_asc/finance/v7+/model.pt'
main_asc = predict_asc.Predict_asc(bert_model, model_pt)

with open('../Test_Data.json') as f:
    data = json.load(f)

eg_df = pd.read_csv('../Submit_Example.csv')

def find_longest(words_list):
    words_list = [w for w in words_list if w]
    words_list = sorted(words_list, key=len)
    longest_words = []
    exclude = []
    for i, w in enumerate(words_list):
        if i in exclude:
            continue
        exclude.append(i)
        longest_word = w
        for j, w2 in enumerate(words_list):
            if j in exclude:
                continue
            if w in w2:
                longest_word = w2
        if longest_word not in longest_words:
            longest_words.append(longest_word)
    return longest_words


result = {'id':eg_df['id'].tolist(), 'negative':[], 'key_entity':[]}
for _id in tqdm(result['id']):
    lines = data.get(_id, [])
    if not lines:
        print('%s 数据异常，判断为非负面'%_id)
        result['negative'].append(0)
        result['key_entity'].append(None)
        continue
    predict_result = main_asc.predict(lines)
    key_et = []
    for label, line in zip(predict_result['label_ids'], lines):
        if label==1:
            key_et.append(''.join(line['term'].split()))
            #key_et.append(''.join(line['raw_term'].split()))
    key_et = list(set(key_et))
    key_et = find_longest(key_et)
    if key_et:
        result['negative'].append(1)
        key_et_join = ';'.join(sorted(key_et))
        result['key_entity'].append(key_et_join)
    else:
        result['negative'].append(0)
        result['key_entity'].append(None)

result_df = pd.DataFrame.from_dict(result)
result_df.to_csv('result.csv', index=False)

