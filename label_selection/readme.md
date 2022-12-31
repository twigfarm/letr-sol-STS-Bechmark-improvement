# Label Selection (remove uncertain pseudo label) 

- 모델 성능 향상을 위한 self-training방식 적용 시 불확실성이 높은 pseudo-label을 선별 
- 사용방법

```python
! git clone https://github.com/tommyEzreal/SolProject3-STS-Bechmark-improvement
%cd ~ /SolProject3-STS-Bechmark-improvement/label_selection/
%run remove_uncertain_label.py
```

```python
# unlabeled sentence pairs
sentence_pairs = [ *unlabeled sentence pairs* ]

# make list of model save path
model_list = [ *model_save_path_1*, *model_save_path_2* , *model_save_path_3* , ... ] # trained model

remove_uncertain_label(model_list,sentence_pairs)

```

## output example
<img width="822" alt="image" src="https://user-images.githubusercontent.com/100064247/210054230-a05ad6d3-5bfb-40f1-9035-9453bf0ddc14.png">
