from pyspark import SparkContext
from pyspark.mllib.feature import HashingTF, IDF

sc = SparkContext(appName="TF-IDF")

documents = sc.textFile("data.txt").map(lambda line: line.split(" "))

hashingTF = HashingTF()
tf = hashingTF.transform(documents)

tf.cache()
idf = IDF().fit(tf)
tfidf = idf.transform(tf)

print("tfidf:")
for each in tfidf.collect():
    print(each)

# idfIgnore = IDF(minDocFreq=1).fit(tf)
# tfidfIgnore = idfIgnore.transform(tf)
# print("tfidfIgnore:")
# for each in tfidfIgnore.collect():
#     print(each)

sc.stop()