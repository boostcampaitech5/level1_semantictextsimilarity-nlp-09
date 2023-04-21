import os
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from hanspell import spell_checker
import math

# 모델의 vocabulary에 추가할 목록 리스트 반환
def get_vocab_data():
    path = "./vocab_data"
    vacobs = os.listdir(path)
    
    result = []
    for v in vocabs:
        df = pd.read_csv(f"./vocab_data/{v}", sep=",", encoding="utf-8", header=None, )
        result.extend(df[0].to_list())      # 단어 리스트만 추가
    
    return result
    
# 두 열의 데이터를 서로 교환(swap)
def data_swap(data_path):
    results = []
    df = pd.read_csv(data_path, sep=",", encoding="utf-8")
    
    # 두 문장의 위치를 수정해 증강된 데이터 저장
    df['sentence_1'], df['sentence_2'] = df['sentence_2'], df['sentence_1']

    # 파일 저장
    df.to_csv(f'{data_path[:-4]}_swap.csv', index=False)

# 여러 분할된 데이터를 하나의 데이터로 붙임(concat)
def data_concat(file_paths, filename):
    contents = []
    
    # concat할 csv 파일을 불러옴
    for fp in file_paths:
        contents.append(pd.read_csv(fp, sep=',', ))
        
    # 붙이고 저장
    df = pd.concat(contents)
    df.to_csv(filename, index=False)
    
def hanspell_translation(src_data, file_path):
    try:
        dasrc_datata = src_data.drop(columns=['id'])
    except:
        pass
    
    data1 = src_data["sentence_1"].to_list()
    data2 = src_data["sentence_2"].to_list()
    
    sent1 = []
    sent2 = []
    
    for i in tqdm(range(len(data))):
        try:
            # 맞춤법 검사시 띄어쓰기도 적용하기 위해 split후 join 진행 ('&'의 경우, hanspell의 오류가 있어 전처리 진행)
            d1 = ''.join(data1[i].replace('&',"").split())
            d2 = ''.join(data2[i].replace('&',"").split())
            
            # 맞춤법 검사 진행
            text1 = spell_checker.check(d1).checked
            text2 = spell_checker.check(d2).checked
            
            # 전처리 진행한 데이터 추가
            sent1.append(text1)
            sent2.append(text2)
            
        except:
            # 만약 오류가 발생한다면 오류 출력 후, 원본 데이터 추가
            print(f"{i+1}'s sentence error {data1[i]} {data2[i]}!!")
            sent1.append(data1[i])
            sent2.append(data2[i])
    
    # 전처리 진행한 데이터를 덮어씀
    data["sentence_1"] = sent1
    data["sentence_2"] = sent2
    
    # print(data[-10:])
    data.to_csv(file_path, index=False)