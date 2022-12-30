import numpy as np
import pandas as pd
import math
import logging
from datetime import datetime

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from sentence_transformers import SentenceTransformer,  LoggingHandler, losses, models, util
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.cross_encoder.evaluation import CECorrelationEvaluator, CEBinaryAccuracyEvaluator, CEBinaryClassificationEvaluator
from sentence_transformers.readers import InputExample
from sentence_transformers import CrossEncoder



from tqdm.auto import tqdm

import seaborn as sns
import matplotlib.pyplot as plt

from importlib import reload
reload(logging)
logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.DEBUG,
    handlers=[LoggingHandler()],
)

logging.info('Run remove_uncertain_label.py')



def remove_uncertain_label (model_list,sentence_pairs):
    
    labels = []
    
    for i,model_save_path in enumerate(model_list):
        
        print(logging.info("Load model_{}".format(i)))
        model = CrossEncoder(model_save_path)
        
        print(logging.info("Predict_with_model_{}".format(i)))
        scores = model.predict(sentence_pairs,show_progress_bar=True)
        
        globals()['label_{}'.format(i)] = np.round((scores*5).tolist(),1)
        labels.append(globals()['label_{}'.format(i)])
        
    
    
    label_mean = []
    label_std = []
    removed_mean = []

    for i in range(len(sentence_pairs)):

        label_each_pair=[]

        for j in range(len(labels)):
            label_each_pair.append(labels[j][i])

        label_std.append(np.round( (np.std(label_each_pair)),2 ) )
        label_mean.append(np.round(sum(label_each_pair)/len(label_each_pair),1))


        min = label_mean[i] - label_std[i]
        max = label_mean[i] + label_std[i]

        for k in label_each_pair:
            if (k < min) | (k > max):
                label_each_pair.remove(k)

        removed_mean.append(sum(label_each_pair) / len(label_each_pair))
        
    return np.round(removed_mean,1)




