import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer,  LoggingHandler, losses, models, util
from sentence_transformers.cross_encoder import CrossEncoder
from tqdm.auto import tqdm




def model_evaluation(model_save_path):
    
    cross_encoder = CrossEncoder(model_save_path)
    
    test_data_path = 'https://raw.githubusercontent.com/tommyEzreal/SolProject3-STS-Bechmark-improvement/main/data/new_devset.csv'
    test = pd.read_csv(test_data_path)
    test = test.rename(columns={'new_label':'labels.label'})
    test = test[['sentence1','sentence2','labels.label']]
    
    # crossencoder의 predict함수 이용하여 새로운 데이터에 대한 prediction 
    pairs_all = list(zip(test['sentence1'], test['sentence2']))
    scores_all = cross_encoder.predict(pairs_all,
                               show_progress_bar = True)

    # 모델 예측값을 silver_label 형태로 달아주기 
    # 소수점 첫째자리 반올림한 라벨로 기입  
    test['silver_label'] = np.round((scores_all * 5).tolist(), 1)
    
    
    # confusion matrix 만들기 위해 3.0을 기준으로 정답여부를 trueN/F - falseN/F 구분하여 데이터 나눠주기 
    correct=[]
    for i in range(len(test)):
      if (test['silver_label'].values[i] >= 3.0) & (test['labels.label'].values[i] >= 3.0):
        correct.append('True_P')
      elif (test['silver_label'].values[i] < 3.0) & (test['labels.label'].values[i] < 3.0):
        correct.append('True_N')
      elif (test['silver_label'].values[i] >= 3.0) & (test['labels.label'].values[i] < 3.0):
        correct.append('False_P')
      else:
        correct.append('False_N')
        
    test['correctness'] = correct
    
    # ACC / F1 score 측정하기 

    TP_all = len(test[test['correctness'] == 'True_P'])
    TN_all = len(test[test['correctness'] == 'True_N'])
    FP_all = len(test[test['correctness'] == 'False_P'])
    FN_all = len(test[test['correctness'] == 'False_N'])
    
    pc_all=TP_all/(TP_all+FP_all)
    rc_all=TP_all/(TP_all+FN_all)

    
    # label bin 만들기 
    a = []
    for i in range(len(test)):
      if (test['labels.label'].values[i] >= 0.0)  & (test['labels.label'].values[i] <=0.2):
        a.append(0.0)
      elif (test['labels.label'].values[i] >= 0.2)  & (test['labels.label'].values[i] <=0.7):
        a.append(0.5)  
      elif (test['labels.label'].values[i] >= 0.8)  & (test['labels.label'].values[i] <=1.2):
        a.append(1.0)
      elif (test['labels.label'].values[i] >= 1.3)  & (test['labels.label'].values[i] <=1.7):
        a.append(1.5)  
      elif (test['labels.label'].values[i] >= 1.8)  & (test['labels.label'].values[i] <=2.2):
        a.append(2.0)
      elif (test['labels.label'].values[i] >= 2.3)  & (test['labels.label'].values[i] <=2.7):
        a.append(2.5)  
      elif (test['labels.label'].values[i] >= 2.8)  & (test['labels.label'].values[i] <=3.2):
        a.append(3.0)
      elif (test['labels.label'].values[i] >= 3.3)  & (test['labels.label'].values[i] <=3.7):
        a.append(3.5)  
      elif (test['labels.label'].values[i] >= 3.8)  & (test['labels.label'].values[i] <=4.2):
        a.append(4.0)
      elif (test['labels.label'].values[i] >= 4.3)  & (test['labels.label'].values[i] <=4.7):
        a.append(4.5)  
      elif (test['labels.label'].values[i] >= 4.8)  & (test['labels.label'].values[i] <=5.0):
        a.append(5.0)

    b = []
    for i in range(len(test)):
      if (test['silver_label'].values[i] >= 0.0)  & (test['silver_label'].values[i] <=0.2):
        b.append(0.0)
      elif (test['silver_label'].values[i] >= 0.3)  & (test['silver_label'].values[i] <=0.7):
        b.append(0.5)  
      elif (test['silver_label'].values[i] >= 0.8)  & (test['silver_label'].values[i] <=1.2):
        b.append(1.0)
      elif (test['silver_label'].values[i] >= 1.3)  & (test['silver_label'].values[i] <=1.7):
        b.append(1.5)  
      elif (test['silver_label'].values[i] >= 1.8)  & (test['silver_label'].values[i] <=2.2):
        b.append(2.0)
      elif (test['silver_label'].values[i] >= 2.3)  & (test['silver_label'].values[i] <=2.7):
        b.append(2.5)  
      elif (test['silver_label'].values[i] >= 2.8)  & (test['silver_label'].values[i] <=3.2):
        b.append(3.0)
      elif (test['silver_label'].values[i] >= 3.3)  & (test['silver_label'].values[i] <=3.7):
        b.append(3.5)  
      elif (test['silver_label'].values[i] >= 3.8)  & (test['silver_label'].values[i] <=4.2):
        b.append(4.0)
      elif (test['silver_label'].values[i] >= 4.3)  & (test['silver_label'].values[i] <=4.7):
        b.append(4.5)  
      elif (test['silver_label'].values[i] >= 4.8)  & (test['silver_label'].values[i] <=5.0):
        b.append(5.0)


    #test

    test['differ']= np.round(test['labels.label'] - test['silver_label'],1)
    test['gold_bin'] = a
    test['silver_bin'] = b

    test['same_bin']= (test['gold_bin'] == test['silver_bin'])

    # bin간의 거리
    test['bin_distance'] = test['gold_bin'] - test['silver_bin']

    #MSE 측정 
    test['SE'] = test['differ']**2 

    # make bin
    bin_05 = test[test['gold_bin'] == 0.5]
    bin_10 = test[test['gold_bin'] == 1.0]
    bin_15 = test[test['gold_bin'] == 1.5]
    bin_20 = test[test['gold_bin'] == 2.0]
    bin_25 = test[test['gold_bin'] == 2.5]
    bin_30 = test[test['gold_bin'] == 3.0]
    bin_35 = test[test['gold_bin'] == 3.5]
    bin_40 = test[test['gold_bin'] == 4.0]
    bin_45 = test[test['gold_bin'] == 4.5]
    bin_50 = test[test['gold_bin'] == 5.0]
    
    # Confusion matrix

    # acc / f1

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
    
    print("[model_save_path]:",model_save_path)
    print("[test_data_path]:",test_data_path)
    print()
    print()
    print("[Whole Test Evaluation]")
    print()
    print("* Pearson correlation:",test.corr(method='pearson')['silver_label'][0])
    print("* Spearman correlation:",test.corr(method='spearman')['silver_label'][0])
    print()
    print("* precision:", TP_all/(TP_all+FP_all))
    print("* recall:", TP_all/(TP_all+FN_all))
    print("* f1 score:", 2*pc_all*rc_all/(pc_all+rc_all))
    print("* acc:", (TP_all + TN_all)/ (TP_all + TN_all + FP_all + FN_all) )
    print()
    print("* MSE:",sum(test['SE']) / len(test))
    print()
    print()
    print("[Test Evaluation Per Label Bin]")
    print()
    print( "len = number of test_data,", "ME = mean_error")
    print()
    print("<bin_0.5>","pearson:",bin_05.corr(method='pearson')['silver_label'][0].round(2),"spaerman:",bin_05.corr(method='spearman')['silver_label'][0].round(2), "len:",len(bin_05), "MSE:",round(sum(bin_05['SE']) / len(bin_05),2),"ME:",round(sum(bin_05['differ']) / len(bin_05),2))
    print("<bin_1.0>","pearson:",bin_10.corr(method='pearson')['silver_label'][0].round(2),"spaerman:",bin_10.corr(method='spearman')['silver_label'][0].round(2), "len:",len(bin_10), "MSE:",round(sum(bin_10['SE']) / len(bin_10),2),"ME:",round(sum(bin_10['differ']) / len(bin_10),2))
    print("<bin_1.5>","pearson:",bin_15.corr(method='pearson')['silver_label'][0].round(2),"spaerman:",bin_15.corr(method='spearman')['silver_label'][0].round(2), "len:",len(bin_15), "MSE:",round(sum(bin_15['SE']) / len(bin_15),2),"ME:",round(sum(bin_15['differ']) / len(bin_15),2))
    print("<bin_2.0>","pearson:",bin_20.corr(method='pearson')['silver_label'][0].round(2),"spaerman:",bin_20.corr(method='spearman')['silver_label'][0].round(2), "len:",len(bin_20), "MSE:",round(sum(bin_20['SE']) / len(bin_20),2),"ME:",round(sum(bin_20['differ']) / len(bin_20),2))
    print("<bin_2.5>","pearson:",bin_25.corr(method='pearson')['silver_label'][0].round(2),"spaerman:",bin_25.corr(method='spearman')['silver_label'][0].round(2), "len:",len(bin_25), "MSE:",round(sum(bin_25['SE']) / len(bin_25),2),"ME:",round(sum(bin_25['differ']) / len(bin_25),2))
    print("<bin_3.0>","pearson:",bin_30.corr(method='pearson')['silver_label'][0].round(2),"spaerman:",bin_30.corr(method='spearman')['silver_label'][0].round(2), "len:",len(bin_30), "MSE:",round(sum(bin_30['SE']) / len(bin_30),2),"ME:",round(sum(bin_30['differ']) / len(bin_30),2))
    print("<bin_3.5>","pearson:",bin_35.corr(method='pearson')['silver_label'][0].round(2),"spaerman:",bin_35.corr(method='spearman')['silver_label'][0].round(2), "len:",len(bin_35), "MSE:",round(sum(bin_35['SE']) / len(bin_35),2),"ME:",round(sum(bin_35['differ']) / len(bin_35),2))
    print("<bin_4.0>","pearson:",bin_40.corr(method='pearson')['silver_label'][0].round(2),"spaerman:",bin_40.corr(method='spearman')['silver_label'][0].round(2), "len:",len(bin_40), "MSE:",round(sum(bin_40['SE']) / len(bin_40),2),"ME:",round(sum(bin_40['differ']) / len(bin_40),2))
    print("<bin_4.5>","pearson:",bin_45.corr(method='pearson')['silver_label'][0].round(2),"spaerman:",bin_45.corr(method='spearman')['silver_label'][0].round(2), "len:",len(bin_45), "MSE:",round(sum(bin_45['SE']) / len(bin_45),2),"ME:",round(sum(bin_45['differ']) / len(bin_45),2))
    print("<bin_5.0>","pearson:",bin_50.corr(method='pearson')['silver_label'][0].round(2),"spaerman:",bin_50.corr(method='spearman')['silver_label'][0].round(2), "len:",len(bin_50), "MSE:",round(sum(bin_50['SE']) / len(bin_50),2),"ME:",round(sum(bin_50['differ']) / len(bin_50),2))
    print()
    print("[Confusion Matrix For Each Label Bin]")
    print()
    print("TP = 유사한페어를 유사하다고 정답,", "TN = 유사하지않은 페어를 유사하지 않다고 정답,", "FP=유사하지 않은페어를 유사하다고 오판,", "FN = 유사한페어를 유사하지않다고 오판")
    print()
    print("* 1.8-2.2:")
    print(confusion_matrix(bin_20))
    print()
    print("* 2.3-2.7:")
    print(confusion_matrix(bin_25))
    print()
    print("* 2.8-3.2:")
    print(confusion_matrix(bin_30))
    print()
    print("* 3.3-3.7:")
    print(confusion_matrix(bin_35))
    print()
    print("* 3.8-4.2:")
    print(confusion_matrix(bin_40))
    print()
    print("* 4.3-4.7:")
    print(confusion_matrix(bin_45))
    print()
    print("* 4.8-5.0:")
    print(confusion_matrix(bin_50))
