from pyspark.ml.feature import HashingTF, IDF, Tokenizer
from pyspark.ml.feature import StopWordsRemover
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("IR").getOrCreate()

sentenceData = spark.createDataFrame([
    ("doc1", "Hi I heard about Spark and I love Spark"),
    ("doc2", "I wish Java could use case classes"),
    ("doc3", "Logistic regression models are neat")
], ["label", "sentence"])

tokenizer = Tokenizer(inputCol="sentence", outputCol="words")
wordsData = tokenizer.transform(sentenceData)
print("wordsData")
wordsData.show(truncate=False)

remover = StopWordsRemover(inputCol="words", outputCol="filtered")
filteredData = remover.transform(wordsData)
print("filteredData")
filteredData.select("words", "filtered").show(truncate=False)

hashingTF = HashingTF(inputCol="filtered", outputCol="rawFeatures", numFeatures=20)
featurizedData = hashingTF.transform(filteredData)
print("featurizedData")
featurizedData.select("filtered", "rawFeatures").show(truncate=False)

idf = IDF(inputCol="rawFeatures", outputCol="features")
idfModel = idf.fit(featurizedData)
rescaledData = idfModel.transform(featurizedData)
print("rescaledData")
rescaledData.select("label", "features").show(truncate=False)
# rescaledData.show(truncate=False)
