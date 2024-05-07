"""Feature engineers the Movies-Credits dataset."""

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
import ast

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


def read_csv(client, bucket_name, file_name, date_cols=None):
    obj = client.get_object(Bucket=bucket_name, Key=file_name) 
    return pd.read_csv(obj['Body'],parse_dates=date_cols)


def get_data(x, cols, dict):
    for col in cols:
        for i in range(len(x[col])):
            for j in range(len(x[col][i])):
                x[col][i][j] = x[col][i][j][dict]
    return x


def preprocess_movie_data(movie):
    movie[['genres','keywords','production_companies','production_countries','spoken_languages']] = movie[['genres','keywords','production_companies','production_countries','spoken_languages']].map(lambda x : ast.literal_eval(str(x)))
    movie = get_data(movie, ['genres','keywords','production_companies','production_countries','spoken_languages'],'name')
    movie[['budget','id','popularity','revenue','runtime','vote_average','vote_count']] = movie[['budget','id','popularity','revenue','runtime','vote_average','vote_count']].apply(pd.to_numeric, errors = 'coerce')
    movie.head(5)
    return movie


def preprocess_credit_data(credit):
    credit[['cast', 'crew']] = credit[['cast', 'crew']].map(lambda x : ast.literal_eval(str(x)))
    credit = get_data(credit, ['cast', 'crew'],'name')
    credit.head(5)
    return credit


if __name__ == "__main__":

    logger.debug("Starting preprocessing.")
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-data-movies", type=str, required=True)
    parser.add_argument("--input-data-credits", type=str, required=True)

    args = parser.parse_args()

    base_dir = "/opt/ml/processing"
    pathlib.Path(f"{base_dir}/data").mkdir(parents=True, exist_ok=True)
    input_data_movies = args.input_data_movies
    input_data_credits = args.input_data_credits

    logger.info("Extracting bucket name and file key from the input_data parameter")
    bucket = 'moviereco'
    key = 'tmdb-movie-metadata/'
    
    logger.info("Downloading data from bucket: %s, key: %s", bucket, key)
    fn_movies = f"{base_dir}/data/tmdb_5000_movies.csv"
    fn_credits = f"{base_dir}/data/tmdb_5000_credits.csv"

    s3 = boto3.resource("s3")
    s3.Bucket(bucket).download_file(key, fn_movies)
    s3.Bucket(bucket).download_file(key, fn_credits)

    logger.debug("Reading downloaded data.")
    # credits_file_name = 'tmdb-movie-metadata/tmdb_5000_credits.csv'
    # movies_file_name = 'tmdb-movie-metadata/tmdb_5000_movies.csv'
    movies_data = read_csv(s3, bucket, fn_movies)
    credits_data = read_csv(s3, bucket, fn_credits)
    
    movie = preprocess_movie_data(movies_data)
    credit = preprocess_credit_data(credits_data)

    movie = pd.merge(movie, credits[['movie_id','cast', 'crew']],  left_on= "id", right_on = "movie_id", how = "left")
    movie['overview'] = movie['overview'].astype(str)

    logger.info("Writing out datasets to %s.", base_dir)
    pd.DataFrame(movie).to_csv(f"{base_dir}/train.csv", header=True, index=True)
