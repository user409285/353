# coding: utf-8
import re
import json
from collections import defaultdict
from multiprocessing import Pool
from tqdm import tqdm

import pandas as pd
from sklearn.model_selection import train_test_split

import jieba
from jieba.analyse import textrank

jieba.dt.tmp_dir = './'
jieba.default_logger.setLevel(jieba.logging.INFO)

def text2sentence(text):
    return re.split('[?？。!！]', text)

def clean_text(text):
    text = re.sub('{IMG:.*?}','',text)
    text = re.sub('\?{3,}','',text)
    return text

def tran2asc(record, tag='train'):
    sentences = []
    if record['title'] == record['text']:
        text = record['text']
        sentences.append(text)
    elif record['title'] and record['text']:
        sentences.append(record['title'])
        sentences.append(record['text'])
    elif record['title'] and not record['text']:
        text = record['title']
        sentences.append(text)
    else:
        text = record['text']
        sentences.append(text)
    sentences = list(set(sentences))
    #sentences = [clean_text(t) for t in sentences]
    entities, key_entities = [], []
    if record['entity']:
        entities = record['entity'].split(';')
    if record.get('key_entity', []):
        key_entities = record['key_entity'].split(';')
    for ke in key_entities:
        if ke not in entities:
            entities.append(ke)
    result = []
    for i, e in enumerate(entities):
        for j, sentence in enumerate(sentences):
            sentence = ''.join(sentence.split())
            #text_join = ' '.join(list(sentence))
            find_e = sentence.find(e)
            if find_e>-1:
                if find_e<50:
                    #text_gram = sentence[:128-find_e+len(e)]
                    text_gram = sentence
                else:
                    start_i = find_e-60 if find_e-60>0 else 0
                    text_gram = sentence[start_i:find_e+len(e)+60]
                text_join = ' '.join(list(text_gram))
                r = {
                        'term': ' '.join(list(e)),
                        'id': record['id']+'_%s%s'%(i,j),
                        'sentence': text_join}
                if e in key_entities:
                    r['polarity'] = 'negative'
                else:
                    r['polarity'] = 'others'
                result.append(r)
            #else:
            #    if tag!='train':
            #        continue
            #    for word in textrank(sentence, topK=None):
            #        find_e = sentence.find(word)
            #        if find_e<50:
            #            text_gram = sentence[:128-find_e+len(e)]
            #        else:
            #            text_gram = sentence[find_e-50:find_e+len(e)+50]
            #        text_join = ' '.join(list(text_gram))
            #        r = {
            #                'polarity': 'others',
            #                'term': ' '.join(list(word)),
            #                'id': record['id']+'_%s%s'%(i,j),
            #                'sentence': text_join}
            #        result.append(r)
            #        break
    return result

def read_csv(file_path):
    data = pd.read_csv(file_path)
    return data

def maptrain(rec):
    rec = rec[1]
    rec_asc = tran2asc(rec)
    return {'id':rec['id'], 'result': rec_asc}

def csv2asc(csv_path):
    df = read_csv(csv_path)
    df = df.fillna('')
    all_rows = list(df.iterrows())
    train_data, dev_data = train_test_split(all_rows, test_size=0.2, random_state=23)
    #train_data = all_rows
    train_result = {}
    with Pool(44) as pool:
        traintrans = pool.map(maptrain, train_data)
    for trans_result in traintrans:
        for r in trans_result['result']:
            train_result[r['id']] = r
    #for rec in train_data:
    #    rec = rec[1]
    #    for r in tran2asc(rec):
    #        train_result[r['id']] = r
    with open('./asc/finance/train.json', 'w') as f:
        json.dump(train_result, f, ensure_ascii=False)

    dev_result = {}
    with Pool(44) as pool:
        devtrans = pool.map(maptrain, dev_data)
    for trans_result in devtrans:
        for r in trans_result['result']:
            dev_result[r['id']] = r
    #for rec in dev_data:
    #    rec = rec[1]
    #    for r in tran2asc(rec):
    #        dev_result[r['id']] = r
    with open('./asc/finance/dev.json', 'w') as f:
        json.dump(dev_result, f, ensure_ascii=False)

def maptest(rec):
    rec = rec[1]
    rec_asc = tran2asc(rec, tag='test')
    if rec_asc:
        return {'id':rec['id'], 'result': rec_asc}
    else:
        return None

def test2asc(testcsv, outputname= './Test_Data.json'):
    df = read_csv(testcsv)
    df = df.fillna('')
    all_rows = list(df.iterrows())
    test_result = defaultdict(list)
    with Pool(44) as pool:
        with tqdm(total=len(all_rows)) as pbar:
            for r in tqdm(pool.imap_unordered(maptest, all_rows)):
                if r:
                    test_result[r['id']] = r['result']
                pbar.update()
    with open(outputname, 'w') as f:
        json.dump(test_result, f, ensure_ascii=False)

if __name__=='__main__':
    csv2asc("Train_Data.csv")
    test2asc("Test_Data.csv")
    test2asc("Train_Data.csv", "pt_train.json")
