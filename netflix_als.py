import json
from distutils.text_file import TextFile
from pyspark import SparkContext
from pyspark.sql.session import SparkSession
from pyspark.sql.types import StructType, StructField
from pyspark.sql.types import FloatType, IntegerType, StringType
from recommenders.datasets.spark_splitters import spark_random_split
from recommenders.evaluation.spark_evaluation import SparkRatingEvaluation, SparkRankingEvaluation
from pyspark.ml.recommendation import ALS
from pyspark.sql.functions import col


sc = SparkContext.getOrCreate()
sc.stop()

""" 
spark = SparkSession.builder \
    .appName("als") \
    .config("spark.driver.memory", "8g").config("spark.executor.memory", "8g").getOrCreate() """

spark = SparkSession \
    .builder \
    .appName("als") \
    .getOrCreate()

sp_rdd = spark.read.json(
    "hdfs://namenode:9000/mydataset/proc_netflix/pre_part-02.json", multiLine=True)

NETFLIX_DATA_SIZE = "500MB"

COL_Review_id = "review_id"
COL_Reviewer = "reviewer"
COL_Movie = "movie"
COL_Rating = "rating"
COL_Review_summary = "review_summary"
COL_Helpful = "helpful"
COL_PREDICTION = "prediction"
schema = StructType(
    (
        StructField(COL_Review_id, IntegerType()),
        StructField(COL_Reviewer, IntegerType()),
        StructField(COL_Movie, StringType()),
        StructField(COL_Rating, IntegerType()),
        StructField(COL_Review_summary, StringType()),
        StructField(COL_Helpful, StringType()),
    )
)

RANK = 100
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
    userCol=COL_Reviewer,
    itemCol=COL_Review_id,
    ratingCol=COL_Rating,
    coldStartStrategy="drop"
)


ml_model = als.fit(data_frames_train)
data_frames_predicate = ml_model.transform(data_frames_test).drop(COL_Rating)

print("evaluating...")
evaluations = SparkRatingEvaluation(
    data_frames_test,  # true labels not tests?
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


rank_eval = SparkRankingEvaluation(data_frames_test, data_frames_predicate, k=100, col_user=COL_Reviewer, col_item=COL_Review_id,
                                   col_rating=COL_Rating, col_prediction=COL_PREDICTION)

print("Model:\tALS",
      "Top K:\t%d" % rank_eval.k,
      "MAP:\t%f" % rank_eval.map_at_k(),
      "NDCG:\t%f" % rank_eval.ndcg_at_k(),
      "Precision@K:\t%f" % rank_eval.precision_at_k(),
      "Recall@K:\t%f" % rank_eval.recall_at_k(), sep='\n')

spark.stop()
