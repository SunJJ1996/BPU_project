import json
import os
# f = open('train.jsonl', 'r', encoding='utf8')
# I = ""
# for i in f.readlines():
#     I = I+i
#
# text = json.loads(I)
# print(len(text))

# 由于文件中有多行，直接读取会出现错误，因此一行一行读取
file = open("train.jsonl", 'r', encoding='utf-8')
papers = []
for line in file.readlines():
    dic = json.loads(line)
    papers.append(dic)

print(papers[0])
print(len(papers))