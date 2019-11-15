# 金融信息负面及主体判定 

比赛链接: https://www.datafountain.cn/competitions/353
参考项目 "[BERT Post-Training for Review Reading Comprehension and Aspect-based Sentiment Analysis](https://github.com/howardhsu/BERT-for-RRC-ABSA)".

用到预训练模型： [RoBERTa](https://github.com/brightmart/roberta_zh)

## 赛题任务 Question Task
```
该任务分为两个子任务：
给定一条金融文本和文本中出现的金融实体列表，

负面信息判定：判定该文本是否包含金融实体的负面信息。如果该文本不包含负面信息，或者包含负面信息但负面信息未涉及到金融实体，则负面信息判定结果为0。

负面主体判定：如果任务1中包含金融实体的负面信息，继续判断负面信息的主体对象是实体列表中的哪些实体。
```
## 思路
```
1、该问题实际为一个文本分类问题，输入：文本段落、实体，输出：是否负面
2、对训练数据做数据探索，实体最终保留的是最长的，label的实体都出现在待判断列表里，所以不用自己再做实体识别
3、难点在于判断文本正负面的目标是不是该实体
4、label不是全部正确，需要剔除这部分噪音
5、训练集不多，且估计选手模型大多大同小异，真正要做的其实是数据增强，经多次尝试后得到目前最优的结果，详细的处理见trans2asc_v7
```
## 执行步骤
```
1、生成post train和finetune格式的语料
> python trans2asc_v7.py 
> python tosquadfmt.py

2、用RoBERTa做post-train，得到post-train后的模型
> cd script
> bash pt.sh finance 5 70000 0 

3、用RoBERTa做finetune，得到finetune后的模型
> cd script
> bash run_absa.sh asc finance_pt finance pt_asc 4 0 

4、对测试集做预测
> cd src
> python runtestdata.py
如有需要，runvaliddata.py是对验证集做预测，得到评估分数
```

## 结果
```
主要是finetune起作用，post-train提升不到0.01，也可能跟本人选用训练语料太小有关（找不到更合适的语料，用原数据改造随便训练一下）

结果文件：src/result.csv  
得分：0.93714833
```
