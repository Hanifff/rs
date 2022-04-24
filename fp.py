from pyspark.ml.fpm import FPGrowth
from numpy import array
from math import sqrt
from pyspark.sql import SparkSession
from pyspark.sql import functions as F


spark = SparkSession \
    .builder \
    .appName("fp") \
    .config("spark.some.config.option", "some-value") \
    .getOrCreate()


# open json file
sc = spark.sparkContext

# A JSON dataset is pointed to by path.
# The path can be either a single text file or a directory storing text files
path="./datasets/netflix/sample.json"
init_df = spark.read.json(path)

# The inferred schema can be visualized using the printSchema() method
df=init_df.drop("review_summary","spoiler_tag","review_detail","helpful").sort("reviewer")
df = df.groupBy("reviewer").agg(F.collect_list("review_id")).sort('reviewer')
df.printSchema()
df.show()

# root
#  |-- movie: string (nullable = true)
#  |-- rating: string (nullable = true)
#  |-- review_date: string (nullable = true)
#  |-- review_id: string (nullable = true)
#  |-- reviewer: string (nullable = true)

# fp-growth
fpGrowth = FPGrowth(itemsCol="collect_list(review_id)",minSupport=0.5, minConfidence=0.6)
model = fpGrowth.fit(df)

# Display frequent itemsets.
model.freqItemsets.show()

# Display generated association rules.
model.associationRules.show()

# transform examines the input items against all the association rules and summarize the
# consequents as prediction
model.transform(df).show()