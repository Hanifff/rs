from distutils.text_file import TextFile
from pickle import DICT
from pyspark.sql import SQLContext, Row
from pyspark import SparkContext
from pyspark.sql.session import SparkSession
from pyspark.sql.types import StructType, StructField
from pyspark.sql.types import FloatType, IntegerType, StringType, NumericType, IntegralType
from recommenders.datasets.spark_splitters import spark_random_split
from recommenders.evaluation.spark_evaluation import SparkRankingEvaluation, SparkRatingEvaluation
from pyspark.ml.recommendation import ALS
from pyspark.ml.feature import StringIndexer
from pyspark.sql.functions import col
import csv
from collections import Counter, defaultdict

# preprocess the data
# find tf for each user from behaviour data set and make a new file
# col:1 user id, col2: news id; col3: user's: tf; col4: news category;
# use it as rating fro a particular newes


behavior = open('../datasets/mind/train/behaviors.tsv')
lines = csv.reader(behavior, delimiter="\t")
news = open('./pre_news.tsv', 'wt')
tsv_writer = csv.writer(news, delimiter='\t')

all_news = open('../datasets/mind/train/news.tsv')
news_lines = csv.reader(all_news, delimiter="\t")

index = {}

for line in lines:
    for terms in line[3]:
        list_news = line[3].split(" ")
        for news_id in list_news:
            tsv_writer.writerow([line[1], news_id, 1])
category = ""
for n_line in news_lines:
    if n_line[0] == news_id:
        category = n_line[1]
        tsv_writer.writerow([line[1], news_id, 1, category])


""" 
sc = SparkContext().getOrCreate()
# sc._conf.get('spark.driver.memory')
spark = SparkSession(sc)

sp_rdd = spark.sparkContext.textFile(
    "hdfs://namenode:9000/datasets/mind/train/news.tsv")

MIND_DATA_SIZE = "1.37GB"  # only behaiviour.tsv

COL_ImpressionID = "ImpressionID"
COL_UserID = "UserID"
COL_Time = "Time"
COL_History = "History"
COL_Impressions = "Impressions"

schema = StructType(
    (
        StructField(COL_ImpressionID, IntegerType()),
        StructField(COL_UserID, FloatType()),
        StructField(COL_Time, FloatType()),
        StructField(COL_History, FloatType()),
        StructField(COL_Impressions, FloatType()),
    )
)

print(sp_rdd.count())

RANK = 3
MAX_ITER = 15
REG_PARAM = 0.05
# get top 10
K = 10

spark.conf.set("spark.sql.analyzer.failAmbiguousSelfJoin", "false")


data_frames_train, data_frames_test = spark_random_split(
    sp_rdd, ratio=0.75, seed=42)

als = ALS(
    maxIter=MAX_ITER,
    rank=RANK,
    regParam=REG_PARAM,
    userCol=COL_UserID,
    itemCol=COL_History,
    ratingCol=COL_History,
    coldStartStrategy="drop"
)


ml_model = als.fit(data_frames_train)

data_frames_predicate = ml_model.transform(data_frames_test).drop(COL_Rating)
print("len data_frames_predicate ", data_frames_predicate.count())

evaluations = SparkRatingEvaluation(
    data_frames_test,
    data_frames_predicate,
    col_user=COL_Reviewer,
    col_item=COL_Review_id,
    col_rating=COL_Rating,
    col_prediction=COL_PREDICTION
)

print(
    "RMSE score = {}".format(evaluations.rmse()),
    "MAE score = {}".format(evaluations.mae()),
    "R2 score = {}".format(evaluations.rsquared()),
    "Explained variance score = {}".format(evaluations.exp_var()),
    sep="\n"
)
 """
