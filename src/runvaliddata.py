import json
from tqdm import tqdm
from collections import defaultdict

import pandas as pd
import predict_asc

bert_model = '../pt_model/finance_pt/'
model_pt = '../run/pt_asc/finance/1/model.pt'
print(bert_model)
print(model_pt)

main_asc = predict_asc.Predict_asc(bert_model, model_pt)

with open('../asc/finance/dev.json') as f:
    devdata = json.load(f)

data = defaultdict(list)
for k, d in devdata.items():
    _id = k.split('_')[0]
    data[_id].append(d)

eg_df = pd.read_csv('../Train_Data.csv')
eg_df = eg_df.fillna('')
eg_df.index = eg_df['id']


result = {'id':list(data.keys()), 'negative':[], 'key_entity':[], 'label_negative':[], 'label_key_entity':[], 'neg_mark':[], 'et_mark':[], 'text':[]}
TPS, FPS, FNS = 0,0,0
TPE, FNE, FPE = 0,0,0

def find_longest(words_list):
    longest_words = []
    exclude = []
    for i, w in enumerate(words_list):
        group = [w]
        if i in exclude:
            continue
        exclude.append(i)
        for j, w2 in enumerate(words_list):
            if j in exclude:
                continue
            if w in w2 or w2 in w:
                group.append(w)
                group.append(w2)
                exclude.append(j)
        longest_word = sorted(group, key=lambda x:len(x), reverse=True)[0]
        longest_words.append(longest_word)
    return longest_words

for _id in tqdm(result['id']):
    lines = data.get(_id, [])
    if not lines:
        print('%s 数据异常，判断为非负面'%_id)
        result['negative'].append(0)
        result['key_entity'].append(None)
        continue
    predict_result = main_asc.predict(lines)
    key_et = []
    label_key_et = []
    for label, line in zip(predict_result['label_ids'], lines):
        this_et = ''.join(line['term'].split())
        #this_et = ''.join(line['raw_term'].split())
        if label==1:
            key_et.append(this_et)
        if line['polarity']=='negative':
            label_key_et.append(this_et)
    key_et = list(set(key_et))
    key_et = find_longest(key_et)
    label_key_et = list(set(label_key_et))
    if key_et:
        result['negative'].append(1)
        key_et_join = ';'.join(key_et)
        result['key_entity'].append(key_et_join)
        if label_key_et:
            TPS += 1
        else:
            FPS += 1
    else:
        if label_key_et:
            FNS +=1 
        result['negative'].append(0)
        result['key_entity'].append(None)

    for ket in key_et:
        if ket in label_key_et:
            TPE += 1
        else:
            FPE += 1
    for lket in label_key_et:
        if lket not in key_et:
            FNE += 1

    if label_key_et:
        result['label_negative'].append(1)
        label_key_et_join = ';'.join(label_key_et)
        result['label_key_entity'].append(label_key_et_join)
    else:
        result['label_negative'].append(0)
        result['label_key_entity'].append(None)

    if len(set(label_key_et)&set(key_et))==len(key_et):
        result['et_mark'].append(True)
    else:
        result['et_mark'].append(False)
    
    lb = 0
    if label_key_et:
        lb = 1
    pred = 0
    if key_et:
        pred = 1
    if lb==pred:
        result['neg_mark'].append(True)
    else:
        result['neg_mark'].append(False)
    df_line = eg_df.loc[_id]
    result['text'].append('|||'.join(list(set([df_line['title'], df_line['text']]))))

PS = TPS/(TPS+FPS)
RS = TPS/(TPS+FNS)
F1S = 2*PS*RS/(PS+RS)

PE = TPE/(TPE+FPE)
RE = TPE/(TPE+FNE)
F1E = 2*PE*RE/(PE+RE)

F1 = 0.4*F1S+0.6*F1E
print('F1s: %s\nF1e: %s\nF1: %s'%(F1S, F1E, F1))
result_df = pd.DataFrame.from_dict(result)
result_df.to_csv('dev_result.csv', index=False)

