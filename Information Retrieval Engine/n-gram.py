
from pyspark.ml.feature import NGram
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName('NGram').getOrCreate()

wordDataFrame = spark.createDataFrame([
    (0, ['Hi', 'I', 'heard', 'about', 'Spark']),
    (1, ['I', 'wish', 'Java', 'could', 'use', 'case', 'classes']),
    (2, ['Logistic', 'regression', 'models', 'are', 'neat'])
], ['id', 'words'])

ngram = NGram(n=3, inputCol='words', outputCol='ngram')

ngram_df = ngram.transform(wordDataFrame)
ngram_df.select("words", "ngram").show(truncate=False)

spark.stop()