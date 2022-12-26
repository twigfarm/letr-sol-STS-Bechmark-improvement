### Model Evaluation

- 새로 제작한 devset을 testset으로 사용하여 STS 모델의 성능을 평가할 수 있습니다. 
- 기존 devset에서 잘못된 라벨들을 수정하고 분포에 맞춰 데이터를 변경하였습니다.
- 피어슨 상관계수, 스피어만 상관계수를 비롯해 confusion_matrix, f1score, mse, 구간별 correlation, acc를 한번에 확인할 수 있습니다. 
- 사용방법
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
