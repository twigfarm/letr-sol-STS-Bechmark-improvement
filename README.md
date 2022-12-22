# SolProject3-STS-bechmark-improvement
2022.09 ~ 2022.12 트위그팜에서 진행한 SolProject3기 인턴쉽 유사문장평가 벤치마크 연구과제 

## 문장유사도 벤치마크 개선 & 모델 성능 향상 
> 기존 STS benchmark의 문제점을 개선하고 개선된 데이터셋을 바탕으로 모델 성능향상 방안 모색 

### 기존 KLUE STSset의 문제점 

- trainset의 균일하지 않은 점수분포 
- 잘못된 번역으로 생성된 데이터들
- 주관적 판단이 개입되는 점수 척도
- 2.0-3.5구간의 라벨오류로 인한 낮은 모델 성능 

### 연구목표

- 기존 STS dataset의 모호한 점수척도 재정립 및 재라벨링 
- 변경된 데이터의 유효성 검증 및 모델 성능 개선 

### 기존 benchmark의 문제점

#### 1. KorSTS

- STS-B 영어 데이터셋을 번역하여 사용 
- 원 데이터의 라벨점수를 그대로 사용했기 때문에 번역으로 인해 발생하는 문맥적 차이를 라벨에 반영하지 못함

#### 2. KLUE-STS

- 균일하지 않은 trainset의 점수분포 
- 잘못된 번역으로 생성된 데이터들 
- 주관적 판단이 개입되는 점수 척도
- binary판단에 대한 Fasle비율이 높은 2.0-3.5구간 

### 연구1: STS annotation 척도 재정립 및 재라벨링

* 라벨 척도의 재정립 

<img width="1329" alt="스크린샷 2022-12-22 오후 5 46 17" src="https://user-images.githubusercontent.com/100064247/209094695-f93411db-558f-4bfb-bd46-343276df7a5b.png">

* 재라벨 대상 데이터 선정 

<img width="1195" alt="image" src="https://user-images.githubusercontent.com/100064247/209095264-b3efbdbb-ae7f-4e59-aaeb-2ac699de377d.png">

* trainset전체에 대해 cross-validation을 진행하여 1차적 추출 

<img width="999" alt="image" src="https://user-images.githubusercontent.com/100064247/209095685-1435b0ca-06e3-4d8d-ac1d-a7b21171066a.png">

* 추출된 데이터들 중 예측불확실성이 높은 데이터와의 교집합 추출 

<img width="1026" alt="image" src="https://user-images.githubusercontent.com/100064247/209095773-0a69de24-f150-450a-84a2-04326248c39a.png">

* devset과 trainset 합해서 500여개의 재라벨 대상데이터에 대해 라벨링 진행 

<img width="801" alt="image" src="https://user-images.githubusercontent.com/100064247/209095965-73aa4b47-3a53-440e-8bfd-6830eb561a7c.png">

### 연구2: 변경된 데이터의 유효성검증, 모델 성능개선 

* 재라벨 이후 새로운 train/val/test set 제작 
  - 3.0미만의 낮은 유사도라벨은 word overlap이 높은 데이터들을 우선적으로 포함, 3.0이상의 높은 유사도 라벨은 word overlap이 낮은 데이터를 우선적으로 포함시켜 모델이 word overlap기반으로 문장유사도를 판단하는 경향성을 반영하지 않도록 testset을 구성 
 
<img width="1225" alt="image" src="https://user-images.githubusercontent.com/100064247/209096382-a8f1feb5-f2cb-451b-add5-749bf8fdb826.png">

#### 변경된 데이터의 유효성 검증 - 모델성능 비교실험

* 모델선정 

- cross-encding vs bi-encoding 

|**Model** trained with|**Pearson R**|**F1 score**|**MSE**|
|------|---|---|---|
|*Bi-encoder* / *KLUE-RoBERTa-base*|||
|original klue-trainset|87.60|80.03|-|
|new trainset|88.08|80.65|-|
|*Cross-encoder* / *KLUE-RoBERTa-base*|||
|original klue-trainset|90.55|81.38|0.41|
|new trainset|91.14|82.98|0.37|




| First Header  | Second Header | Third Header         |
| :------------ | :-----------: | -------------------: |
| First row     | Data          | Very long data entry |
| Second row    | **Cell**      | *Cell*               |
| Third row     | Cell that spans across two columns  ||
[Table caption, works as a reference][section-mmd-tables-table1]

| A | B | C |
| <r2> 1 | <c2> 2 |
| 가 | 나 |

