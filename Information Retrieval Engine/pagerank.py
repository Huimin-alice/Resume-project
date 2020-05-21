import re
from operator import add
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName('Page Rank').getOrCreate()

lines = spark.read.text('data.txt').rdd.map(lambda r: r[0])
# 1 2
# 1 ->  2

def parseNeighbors(urls):
    # 1 2
    parts = re.split(r'\s+', urls)
    return parts[0], parts[1] #(1 , 2)

links = lines.map(parseNeighbors).distinct().groupByKey().cache()
# url,  neighbour url
# url -->  neighbour url

ranks = links.map(lambda url_neighbors: (url_neighbors[0], 1.0))
# 1 -> 2 , 3, 4
#  (1 -> 2 , 3, 4 , 1.0)

def computeContribute(urls, ranks):
    num_urls = len(urls)
    for url in urls:
        yield (url, ranks / num_urls)


iteration = 20
for i in range(iteration):
    contribs = links.join(ranks).flatMap(lambda url_urls_rank: computeContribute(url_urls_rank[1][0], url_urls_rank[1][1]))
    ranks = contribs.reduceByKey(add).mapValues(lambda rank: rank * 0.85 + 0.15)

for (link, rank) in ranks.collect():
    print("%s has rank: %s." % (link, rank))

spark.stop()
