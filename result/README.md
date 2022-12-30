# README

## 파일 설명

---

`klue-sts-v.sol_train`

klue-sts-v1.1_train에서 2.0 ~ 3.5 구간 중 confusion_matrix의 FP, FN에 해당하는 데이터들을 선별하여 재라벨링을 실시한 data set

`klue-sts-v.sol_dev`

klue-sts-v1.1_dev에서 2.0 ~ 3.5 구간 중 재라벨링을 실시한 data set

## guid

---

기존 klue-sts-train 데이터셋 파일에서 사용하는 index

- 회색의 셀은 최종적으로 새로운 라벨이 사용된 데이터임을 의미함.

## source

---

문장의 출처

## sentence1, sentence2

---

쌍을 이루는 문장쌍

## sentence2_retranslation

---

sentence2의 번역이 잘못되었거나, 올바르지 않은 문장이라고 판단된 경우 기계번역을 통해 생성해준 올바른 문장

- sentence2_retransaltion에 null값을 가진 문장쌍은 올바른 문장으로 판단하고, 재번역을 하지 않은 문장임.
- 재번역을 실시한 문장이 존재한다면, 기존의 sentence2는 배재하고, ‘**sentence1’ - ‘sentece2_retranslation’** 간 라벨 점수를 매김.

## label_origin

---

기존 klue-sts-train 데이터셋에서 해당 문장쌍에 부여한 라벨 점수

- 해당 데이터는 재번역이 실시 되기 전 부여된 라벨 점수로,  sentence2_retranslation의 내용과 상관없이 **‘sentnece1’ - ‘sentence2’** 간 비교만으로 라벨 점수가 매겨진 것임.

## origin_bin

---

여러 작업자의 라벨을 평균으로 내어 연속형으로 분포하던 기존 라벨(label_origin)을 새로 정립한 라벨 척도에 맞는 범주로 이산화한 값

| label_origin | origin_bin |
| --- | --- |
| 2.0 / 2.1 / 2.2 | 2.0 |
| 2.3 / 2.4 / 2.5 / 2.6 / 2.7 | 2.5 |
| 2.8 / 2.9 / 3.0 / 3.1 / 3.2 | 3.0 |
| 3.3 / 3.4 / 3.5 | 3.5 |

ex > 기존 라벨 점수(label_origin)가 2.3점에 해당한다면, 2.5에 해당하는 값으로 봄.

## label_new

---

해당 행의 문장쌍 데이터에 새롭게 부여한 최종 점수

- 연한 노란색의 셀은 작업자들 간 점수가 통일되지 않는 문장쌍에 두 작업자의 평균 점수를 부여한 것임.

<aside>
✅ **라벨 교정 기준**
label_origin / origin_bin / label_new를 비교하여 최종적으로 교정해줄 라벨을 선정

ex> 기존 라벨이 2.1이라면 2점 구간에 해당하는 값이므로, 재라벨링 값으로 2.0을 받은 데이터는 교정 해주지 않는다.

| label_origin | origin_bin | label_new | 교정 유무 |
| --- | --- | --- | --- |
| 2.1 | 2 | 2.0 | x |
| 2.3 | 2.5 | 2.5 | x |
| 2.9 | 3 | 3.0 | x |
| 3.3 | 3.5 | 3.0 | o |
| 2.7 | 2.5 | 3.0 | o |
| 2.8 | 3 | 3.5 | o |
</aside>

## label_annotator#

---

개별 작업자가 해당 문장쌍에 부여한 새로운 라벨 점수

- # : 작업자(annotator) 번호
- **ex >** label_annotator 2 : 작업자 2가 부여한 점수

## label_maincontext_annotator #

---

개별 작업자가 해당 문장의 핵심맥락에 부여한 함의 여부

- 2 : 핵심맥락 상호함의
- 1 : 핵심맥락 일방함의
- 0 : 핵심맥락 비함의

## label_subcontext_annotator #

---

개별 작업자가 해당 문장의 보조맥락에 부여한 함의 여부

- 2 : 보조맥락 상호함의
- 1 : 보조맥락 일방함의
- 0 : 보조맥락 비함의

## correctness

---

모델이 오판한 데이터의 confusion-matrix 내 유형

- False_P : 실제값은 False이나 True로 예측 (=3점 이하의 문장이나 3점 이상으로 예측)
- False_N : 실제값은 True이나 False로 예측 (=3점 이상의 문장이나 3점 이하로 예측)

## Summary

---

|  | Train | Dev |
| --- | --- | --- |
| 기존 데이터의 수 |  |  |
| 사용한 데이터의 수 | 412 | 163 |
| 변경된 라벨의 수 | 267 | 91 |
| 재번역한 데이터의 수 | 36 | 6 |