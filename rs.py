from pyspark.sql import SQLContext, Row
from pyspark import SparkContext
from pyspark.sql.session import SparkSession
sc = SparkContext.getOrCreate()
spark = SparkSession(sc)

sp_rdd = spark.sparkContext.textFile(
    "hdfs://namenode:9000/datasets/mind/train/news.tsv")

print(sp_rdd.count())
print(sp_rdd.take(5))


# About mind data set
# https://msnews.github.io/
# https://www.kaggle.com/datasets/arashnic/mind-news-dataset ** PS: need account
# https://github.com/microsoft/recommenders
# https://paperswithcode.com/dataset/mind
# format of file : https://github.com/msnews/msnews.github.io/blob/master/assets/doc/introduction.md
""" behaviors.tsv
The behaviors.tsv file contains the impression logs and users' news click histories. It has 5 columns divided by the tab symbol:

Impression ID. The ID of an impression.
User ID. The anonymous ID of a user.
Time. The impression time with format "MM/DD/YYYY HH:MM:SS AM/PM".
History. The news click history (ID list of clicked news) of this user before this impression. The clicked news articles are ordered by time.
Impressions. List of news displayed in this impression and user's click behaviors on them (1 for click and 0 for non-click). The orders of news in a impressions have been shuffled.
news.tsv
The docs.tsv contains the detailed information of news articles involved in the behaviors.tsv file. It has 7 columns, which are divided by the tab symbol:

News ID
Category
SubCategory
Title
Abstract
URL
Title Entities (entities contained in the title of this news)
Abstract Entities (entites contained in the abstract of this news)
entityembedding.vec & relationembedding.vec
The entityembedding.vec and relationembedding.vec files contain the 100-dimensional embeddings of the entities and relations learned from the subgraph (from WikiData knowledge graph) by TransE method. 
In both files, the first column is the ID of entity/relation, and the other columns are the embedding vector values. We hope this data can facilitate the research of knowledge-aware news 
recommendation. """
