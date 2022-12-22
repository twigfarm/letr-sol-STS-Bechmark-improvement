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

### 연구방향1: STS annotation 척도 재정립 및 재라벨링

* 라벨 척도의 재정립 

<img width="1329" alt="스크린샷 2022-12-22 오후 5 46 17" src="https://user-images.githubusercontent.com/100064247/209094695-f93411db-558f-4bfb-bd46-343276df7a5b.png">

* 재라벨 대상 데이터 선정 

<img width="1195" alt="image" src="https://user-images.githubusercontent.com/100064247/209095264-b3efbdbb-ae7f-4e59-aaeb-2ac699de377d.png">


