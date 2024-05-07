import argparse
import logging
import os
import pathlib
import requests
import tempfile

import boto3
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from io import StringIO
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sentence_transformers import SentenceTransformer, util
import pandas as pd

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

if __name__ == "__main__":


    train_path = "/opt/ml/processing/train.csv"
    movie = pd.read_csv(train_path, header=True)

    def recommender(movie_name):
        result = pd.concat([movie["original_title"],
                        pd.DataFrame(overview_cos_sim[:,movie[movie["original_title"] == movie_name].index].numpy(), columns=['Overview'])],axis = 1)
        result = result[result["Overview"] != 1]
        result = result.sort_values('Overview', ascending= False).head(10).reset_index(drop =  True)
        return result
    
    model = SentenceTransformer('all-MiniLM-L6-v2')
    overview_embeddings = model.encode(movie['overview'])
    overview_cos_sim = util.cos_sim(overview_embeddings, overview_embeddings)
    
