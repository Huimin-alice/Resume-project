from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("Tokenizer").getOrCreate()

sentenceDataFrame = spark.createDataFrame([
    (0, "Hi I heard about Spark and I love Spark"),
    (1, "I wish Java could use case classes"),
    (2, "Logistic,regression,models,are,neat")
], ["id", "sentence"])

from pyspark.sql.types import IntegerType
from pyspark.sql.functions import udf
countTokens = udf(lambda words: len(words), IntegerType())

from pyspark.ml.feature import Tokenizer
from pyspark.sql.functions import col
tokenizer = Tokenizer(inputCol="sentence", outputCol="words")
tokenized = tokenizer.transform(sentenceDataFrame)
print("tokenizer")
tokenized.select("sentence", "words").withColumn("tokens", countTokens(col("words"))).show(truncate=False)

from pyspark.ml.feature import RegexTokenizer
regexTokenizer = RegexTokenizer(inputCol="sentence", outputCol="words", pattern="\\W")
regexTokenized = regexTokenizer.transform(sentenceDataFrame)
print("regex tokenizer")
regexTokenized.select("sentence", "words") .withColumn("tokens", countTokens(col("words"))).show(truncate=False)

spark.stop()