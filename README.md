# [SolProject 3기] STS 문장유사도평가
2022.10 ~ 2022.12 트위그팜에서 진행한 SolProject3기 인턴쉽 유사문장평가 벤치마크 연구과제 
<br/>
<br/>

## 한국어 문장유사도 벤치마크 개선 & 모델 성능 향상 
> 기존 STS benchmark의 문제점을 개선하고 개선된 데이터셋을 바탕으로 모델 성능향상 방안 모색 

<br/>

## Requirements


- `torch ~=1.8.1`
- `transformers ~=4.25.1`
- `sentence-transformers ~=2.2.2`
- `datasets ~=2.7.1`


<br/>

## 연구과제 

- 기존 STS dataset의 모호한 점수척도 재정립 및 재라벨링 
- 변경된 데이터의 유효성 검증 및 모델 성능 개선 
<br/>

## 기존 benchmark의 문제점

#### 1. KorSTS

- STS-B 영어 데이터셋을 번역하여 사용 
- 원 데이터의 라벨점수를 그대로 사용했기 때문에 번역으로 인해 발생하는 문맥적 차이를 라벨에 반영하지 못함

#### 2. KLUE-STS

- 균일하지 않은 trainset의 점수분포 
- 잘못된 번역으로 생성된 데이터들 
- 주관적 판단이 개입되는 점수 척도
- binary판단에 대한 Fasle비율이 높은 2.0-3.5구간 
<br/>

## 연구1: STS annotation 척도 재정립 및 재라벨링

#### 라벨 척도의 재정립 

<img width="1329" alt="스크린샷 2022-12-22 오후 5 46 17" src="https://user-images.githubusercontent.com/100064247/209094695-f93411db-558f-4bfb-bd46-343276df7a5b.png">
<br/>

#### 재라벨 대상 데이터 선정 

<img width="1195" alt="image" src="https://user-images.githubusercontent.com/100064247/209095264-b3efbdbb-ae7f-4e59-aaeb-2ac699de377d.png">
<br/>

#### trainset전체에 대해 cross-validation을 진행하여 1차적 추출 

<img width="999" alt="image" src="https://user-images.githubusercontent.com/100064247/209095685-1435b0ca-06e3-4d8d-ac1d-a7b21171066a.png">
<br/>

#### 추출된 데이터들 중 예측불확실성이 높은 데이터와의 교집합 추출 

<img width="1026" alt="image" src="https://user-images.githubusercontent.com/100064247/209095773-0a69de24-f150-450a-84a2-04326248c39a.png">
<br/>

#### devset과 trainset 합해서 500여개의 재라벨 대상데이터에 대해 라벨링 진행 

<img width="801" alt="image" src="https://user-images.githubusercontent.com/100064247/209095965-73aa4b47-3a53-440e-8bfd-6830eb561a7c.png">
<br/>

## 연구2: 변경된 데이터의 유효성검증, 모델 성능개선 

* 재라벨 이후 새로운 train/val/test set 제작 
  - 3.0미만의 낮은 유사도라벨은 word overlap이 높은 데이터들을 우선적으로 포함, 3.0이상의 높은 유사도 라벨은 word overlap이 낮은 데이터를 우선적으로 포함시켜 모델이 word overlap기반으로 문장유사도를 판단하는 경향성을 반영하지 않도록 testset을 구성 
 
<img width="1225" alt="image" src="https://user-images.githubusercontent.com/100064247/209096382-a8f1feb5-f2cb-451b-add5-749bf8fdb826.png">
<br/>



### Evaluate model-performance with new_devset


```python
! git clone https://github.com/tommyEzreal/SolProject3-STS-Bechmark-improvement
%cd /SolProject3-STS-Bechmark-improvement/model_evaluation/
%run model_evalutation.py
```
```python
# 학습완료된 model 불러오기 (cross-encoder)
model_save_path = ' '
model_evaluation(model_save_path ,
                 encoding ='cross-encoding')

# bi-encoder
model_save_path = ' '
model_evaluation(model_save_path ,
                 encoding ='bi-encoding')
```
<br/>

### 변경된 데이터의 유효성 검증 - 모델성능 비교실험

#### F1-score의 측정 

```python
def confusion_matrix(test):

  TP = len(test[test['correctness'] == 'True_P'])
  TN = len(test[test['correctness'] == 'True_N'])
  FP = len(test[test['correctness'] == 'False_P'])
  FN = len(test[test['correctness'] == 'False_N'])
  print("TP:",TP,"TN:",TN,"FP:",FP,"FN",FN)

  
  if (TP + FP) & (TP+FN) ==0:
    return("accuracy:",(TP+TN)/(TP+TN+FP+FN))
  
  else:
    pc=TP/(TP+FP)
    rc=TP/(TP+FN)
    
    return("accuracy:",(TP+TN)/(TP+TN+FP+FN),"precision:", TP/(TP+FP), "recall:", TP/(TP+FN),"f1 score:", 2*pc*rc/(pc+rc))
```

#### <모델선정>   
*all results are trained with 10random_seed and 3 epochs


- cross-encding vs bi-encoding 

|**Model** trained with|**Pearson R**|**F1 score**|**MSE**|
|------|---|---|---|
|***Bi-encoder* / *KLUE-RoBERTa-base***|||
|original klue-trainset|87.60|80.03|-|
|new trainset|88.08|80.65|-|
|***Cross-encoder* / *KLUE-RoBERTa-base***|||
|original klue-trainset|90.55|81.38|0.41|
|new trainset|**91.14**|**82.98**|**0.37**|


<br/>

- BERT / ELECTRA / RoBERTa (cross-encoder)

|**Model** trained with|**Pearson R**|**F1 score**|**MSE**|
|------|---|---|---|
|***KLUE-BERT-base***|||
|original klue-trainset|88.32|81.09|0.44|
|new trainset|89.62|80.30|0.41|
|***KoELECTRA-base***|||
|original klue-trainset|88.72|81.12|0.40|
|new trainset|90.54|**83.12**|**0.37**|
|***KLUE-RoBERTa-base***|||
|original klue-trainset|90.55|81.38|0.41|
|new trainset|**91.14**|82.98|**0.37**|

<br/>

## 변경된 데이터의 유효성 검증 - 유효성검증 실험설계 

<img width="1191" alt="image" src="https://user-images.githubusercontent.com/100064247/209103925-5fbecb1f-1980-4fb9-bf6e-8d5d81cfbd0c.png">

<br/>

### 유효성검증 실험결과 

|**Model** trained with|**Pearson R**|**F1 score**|**MSE**|**2.0-3.5<br/>Pearson R**|**2.0-3.5<br/>ACC**|
|---------|---|---|---|----|---|
|***KLUE-BERT-base***|||
|original klue-trainset|90.55|81.38|0.41|39.42|59.36|
|new trainset|**91.14**|**82.98**|**0.37**|**43.73**|**62.68**|


> 전체구간 성능상승 / 재라벨링 타겟구간이었던 2.0-3.5구간의 피어슨계수, acc 더 큰폭으로 상승 

<br/>

#### 모델 성능개선 - semi-supervised learning 

<img width="1245" alt="image" src="https://user-images.githubusercontent.com/100064247/209106383-5998ba70-a001-4b3f-b3bc-2b9439070ce9.png">

#### 모델 성능개선 - data augmentation
<img width="1265" alt="image" src="https://user-images.githubusercontent.com/100064247/209106580-85894063-31de-43ca-82f0-705fd120b418.png">

#### 모델 성능개선 - augmentation range selection
<img width="1250" alt="image" src="https://user-images.githubusercontent.com/100064247/209106741-13d537de-ed67-4613-b02c-4e233d9892dc.png">

#### 모델 성능개선 - pseudo-label selection
<img width="1142" alt="image" src="https://user-images.githubusercontent.com/100064247/209107009-acc1b33d-ed9f-4993-a24b-72fb3b7d404c.png">

<br/>

### 모델 성능개선 실험결과 
|**Model** trained with|**Pearson R**|**F1 score**|**1.8-3.2 <br/>Pearson R**|**Data<br/>Augmentation**|**Num of<br/> Augmented data**|**Aug range**|
|---------|---|---|---|----|---|---|
|***without data augmentation***||||
|original klue-trainset|90.55|81.38|39.45|x|x|x|
|new trainset|91.14|82.65|41.75|x|x|x|
|***with data augmentation***||||
|best-model predicted label|91.61|83.57|43.59|+korsts-test|485|1.8-3.2|
|ensembled label|91.98|83.24|46.44|+korsts-test|485|1.8-3.2|
|best-model predicted label|91.57|**84.04**|46.60|+korsts-test<br/>& paraKAIST|851|1.8-3.2|
|ensembled label|**92.05**|83.30|**49.53**|+korsts-test<br/>& paraKAIST|847|1.8-3.2|
> 한가지 데이터에 대해서만 증강 / 수를 늘려 두가지 모두에 대해 증강한 경우 <br/> 두가지 모두 증강전보다 성능향상 <br/> 앙상블을 통해 불확실성이 높은 라벨을 제거한후 데이터증강을 적용한 경우가 더 높은 성능 

<br/>

## 연구 의의 및 한계점 

#### 의의 
-  라벨링 난이도가 높은 점수구간의 명확한 라벨링 기준 정립
-  양질의 STS라벨 데이터 제작
-  적은수의 라벨교정과 데이터 증강만으로 유의미한 성능변화 도출
-  기존 벤치마크의 문제점을 교정한 새로운 벤치마크로의 발전 가능성
-  모델 성능향상과 데이터 교정 및 증강을 위한 다양한 방법론의 시도, 유의미한 실험결과 도출 

#### 한계점
- 적은 양의 재라벨 데이터 
- 라벨링 가이드의 난이도 상승 
- 해결하지 못한 워드 오버랩기반 예측경향 
- 여전히 불균형한 훈련데이터 



<br/>

## Reference

### Papers
- Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks(2019)
- Augmented SBERT: Data Augmentation Method for Improving Bi-Encoders for Pairwise Sentence Scoring Tasks(2020)
- KLUE: Korean Language Understanding Evaluation(2021)
- KorNLI and KorSTS: New Benchmark Datasets for Korean Natural Language Understanding(2020)
- SemEval2017 Task 1: Semantic Textual Similarity Multilingual and Crosslingual Focused Evaluation
- A survey on semi-supervised learning(2020)
- Learning with pseudo-ensembles(2014)
- Pseudo-Labels Are All You Need(2022)
- 2021년 유사 문장 생성 말뭉치 연구 분석 사업
- 유사말뭉치 분석을 통한 유사도 인식에 관한 연구(2021)

### Others
- https://www.datacamp.com/tutorial/active-learning
- https://blog.est.ai/2020/11/ssl/
- https://github.com/UKPLab/sentence-transformers
- https://github.com/Huffon/klue-transformers-tutorial/blob/master/sentence_transformers.ipynb
- https://pseudo-lab.github.io/klue-baseline/docs/Semantic%20Textual%20Similarity.html
- https://www.youtube.com/watch?v=WS1uVMGhlWQ
- https://www.youtube.com/watch?v=aSx0jg9ZILo
