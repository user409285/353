import json
from tqdm import tqdm
import pandas as pd
from collections import defaultdict
from sklearn.model_selection import train_test_split


with open('pt_train.json') as f:
    data = json.load(f)

eg_df = pd.read_csv('Train_Data.csv')
eg_df = eg_df.fillna('')
eg_df.index = eg_df['id']

all_rows = list(eg_df.iterrows())
train_data, dev_data = train_test_split(all_rows, test_size=0.2)

def run(all_rows, outname = 'out'):
    alldata = {'data': [], 'version': '1.1'}
    
    for rec in tqdm(all_rows):
        _id = rec[0]
        if _id not in data:
            continue
        record = rec[1]
        title,text = '', ''
        if record['title'] == record['text']:
            text = record['text']
            title = record['title']
        elif record['title'] and record['text']:
            text = record['text']
            title = record['title']
        elif record['title'] and not record['text']:
            text = record['title']
            title = record['title']
        else:
            text = record['text']
        text = ' '.join(list(''.join(text.split())))
        result = {}
        result['title'] = ' '.join(list(''.join(title.split())))
        paras = defaultdict(list)
        for line in data[_id]:
            qa = {'answers': [{'answer_start': line['sentence'].find(line['term']), 'text': line['term']}],
                'question': '负' if line['polarity']=='negative' else '无',
                'id': line['id']}
            paras[line['sentence']].append(qa)
        paragraphs = []
        for para, qas in paras.items():
            paragraph = {'context': para, 'qas': qas}
            paragraphs.append(paragraph)
        result['paragraphs'] = paragraphs
        alldata['data'].append(result)
    
    with open('./squad/%s.json'%outname, 'w') as f:
        json.dump(alldata, f, ensure_ascii=False)

run(train_data, 'train')
run(dev_data, 'dev')
