import urllib.request
import urllib
import json
from googletrans import Translator

import pandas as pd
import time
import tqdm

def my_papago(input_text, src, tgt):
    # 키값 불러오기
    # client = { client_id, client_secret }
    with open("keys.json", "r") as f:
        client = json.load(f)

    url = "https://openapi.naver.com/v1/papago/n2mt"

    # 입력할 텍스트 데이터

    encText = urllib.parse.quote(input_text, encoding="utf-8")

    # 데이터 요청
    data = f"source={src}&target={tgt}&text={encText}"
    request = urllib.request.Request(url)

    # PAPAGO API 계정 로그린
    request.add_header("X-Naver-Client-Id",
                       client["papago_api_key"]["client_id"])
    request.add_header("X-Naver-Client-Secret",
                       client["papago_api_key"]["client_secret"])

    # url 호출
    response = urllib.request.urlopen(request, data=data.encode("utf-8"))
    rescode = response.getcode()

    # API 호출 결과
    if (rescode == 200):
        response_body = response.read()
        data = json.loads(response_body)

        translated_text = data["message"]["result"]["translatedText"]
    # 에러 발생
    else:
        print("Error Code:" + rescode)

    time.sleep(1)
    return translated_text
  
def google_trans(texts, src, tgt):
    translator = Translator()
    target = translator.translate(texts, src=src, dest=tgt)
    
    return target.text
  
if __name__ == "__main__":
    # 데이터 로드
    data = pd.read_csv('./EDA/train.csv')
    
    languages = {"english":"en", "japanese":'ja', "mogolian" : "mn", "turkish" : "tr", "hungary":"hu"}
    
    lang = input()
    try:
        start_time = time.time()
        language, token = lang, languages[lang]

        print('\n\n',"="*100)
        print(f"{language} 언어 작업중...")
        sent1, sent2 = [], []
        idx = 0

        for s1, s2 in zip(data['sentence_1'], data['sentence_2']):
            print(f"{idx+1} / {len(data)}")
            idx += 1

            ss1, ss2 = google_trans(s1, "ko", token), google_trans(s2, "ko", token)
            sss1, sss2 = google_trans(ss1, token, "ko"), google_trans(ss2, token, "ko")
            time.sleep(1)
            sent1.append(sss1)
            sent2.append(sss2)
    # 증강된 데이터 저장
    except:
        data['sentence_1'] = sent1
        data['sentence_2'] = sent2

        end_time = time.time()
        print("소요시간:", end_time - start_time)
        data.to_csv(f'./EDA/train_{language}.csv', index=False)