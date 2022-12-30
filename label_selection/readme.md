# Label Selection (remove uncertain pseudo label) 

- 모델 성능 향상을 위한 self-training방식 적용 시 불확실성이 높은 pseudo-label을 선별 
- 사용방법

```python
! git clone https://github.com/tommyEzreal/SolProject3-STS-Bechmark-improvement
%cd ~ /SolProject3-STS-Bechmark-improvement/label_selection/
%run remove_uncertain_label.py
```

```python
sentence_pairs = [ *unlabeled sentence pairs* ]
model_list = [ *model_save_path_1*, *model_save_path_2* , *model_save_path_3* , ... ] # trained model

remove_uncertain_label(model_list,sentence_pairs)

```
