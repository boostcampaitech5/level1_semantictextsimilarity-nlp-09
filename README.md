# Team : 974981

## Project 💡
문맥적 유사도 측정(STS:Semantic textual similarity)
- STS 데이터셋을 활용하여 두 텍스트가 얼마나 유사한지 판단하는 NLP Task
- 일반적으로 두 개의 문장을 입력하고, 이러한 문장쌍이 얼마나 의미적으로 서로 유사한지를 판단
<img src="https://user-images.githubusercontent.com/79522982/230921423-9add08cb-9d2d-490e-a93f-67b20b59eee7.png" width="720px;" height="480px;" alt=""/>  

- 유사도 점수와 함께 두 문장의 유사함을 참과 거짓으로 판단하는 참고 정보도 같이 제공하지만, 최종적으로 0과 5사이의 유사도 점수를 예측하는 것을 목적  
- 아래는 각 데이터의 개수와 Label 점수의 의미입니다.
- 총 데이터 개수 : 10,974 문장 쌍
  - Train(학습) 데이터 개수: 9,324
  - Test(평가) 데이터 개수: 1,100
  - Dev(검증) 데이터 개수: 550
  - Label 점수: 0 ~ 5사이의 실수
    - 5점 : 두 문장의 핵심 내용이 동일하며, 부가적인 내용들도 동일함
    - 4점 : 두 문장의 핵심 내용이 동등하며, 부가적인 내용에서는 미미한 차이가 있음
    - 3점 : 두 문장의 핵심 내용은 대략적으로 동등하지만, 부가적인 내용에 무시하기 어려운 차이가 있음
    - 2점 : 두 문장의 핵심 내용은 동등하지 않지만, 몇 가지 부가적인 내용을 공유함
    - 1점 : 두 문장의 핵심 내용은 동등하지 않지만, 비슷한 주제를 다루고 있음
    - 0점 : 두 문장의 핵심 내용이 동등하지 않고, 부가적인 내용에서도 공통점이 없음
- 각 데이터별 Label 점수는 여러명의 사람이 위의 점수 기준을 토대로 평가한 두 문장간의 점수를 평균낸 값

## Members ✨
<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tr>
    <td align="center"><a href="https://github.com/Broco98"><img src="image/Profile_T5072.png" width="160px;" height="240px;" alt=""/><br /><sub><b>김효연</b>
    <td align="center"><a href="https://github.com/a-Tachyon"><img src="image/Profile_T5107.png" width="160px;" height="240px;" alt=""/><br /><sub><b>서유현</b>
    <td align="center"><a href="https://github.com/MuHyeonSon"><img src="image/Profile_T5114.png" width="160px;" height="240px;" alt=""/><br /><sub><b>손무현</b>
    <td align="center"><a href="https://github.com/MonteCarlolee"><img src="image/Profile_T5144.png" width="160px;" height="240px;" alt=""/><br /><sub><b>이승진</b>
    <td align="center"><a href="https://github.com/Jiwonii97"><img src="image/Profile_T5231.png" width="160px;" height="240px;" alt=""/><br /><sub><b>황지원</b>
  </tr>
</table>
      👉 이름을 클릭하시면 해당 멤버의 <code>Github 페이지</code>로 넘어갑니다
