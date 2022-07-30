from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
from pyspark.sql.functions import lower, col
import sys
import re
import time

start_time = time.time()
#Grab input and output files from parameters
infile = (sys.argv[1])
outfile = (sys.argv[2])

#configure SparkSession
spark = SparkSession.builder.appName("pagerank").getOrCreate()

# Used to get contributions for each link based off of the values of the links RDD
# and the values from the ranks RDD
def calc_contributions(links, ranks):
    for dest in links:
        yield (dest, ranks/ len(links))

#This function assigns 0 to links that were not  in the contributions RDD
def assign_zero(contribution):
    if contribution:
        return contribution
    return 0.0

#Both of these functions are for file formatting per the instructions in the README
#The first lowercases all input lines 
#The second removes lines that have colons unless they start with "category:"
def file_formating(lines):
    if lines:
        lines = lines.lower()
    return lines

def filter_colon(lines):
    colon = ":"
    cat_colon = "category:"
    #this may be wrong, Check if lines can be single value.
    if not lines[1]:
        return False
    if not lines[0]:
        return False
    if colon in lines[0] or colon in lines[1]:
        if cat_colon in lines[0] or cat_colon in lines[1]:
            return True
        return False
    return True

#Reads in a tab seperated list file and removes comments, then stores the file as an RDD
#Then the RDD is set to lowercase and links with colons in them are filtered out
lines = spark.read.format("csv").option("delimiter", "\t").option("comment", "#").load(infile).rdd #Load graph as an RDD of URL, outlinks) pairs
lines = lines.map(lambda x: (file_formating(x[0]), file_formating(x[1])))
lines = lines.filter(lambda x: filter_colon(x)).persist()

#Creates an RDD that stores the link as a key and the outlinks that it references as a list of values.
links = lines.map(lambda urls: urls).distinct().groupByKey().persist()

#Creates an RDD that stores the link as a key and assigns the rank to 1 for the values
ranks = links.map(lambda url: (url[0], 1.0)).persist() #rdd of (URL, Rank) pairs

# contributions are mapped using the calc_contributions function. The function is passed in the values from the links and ranks rdd
# the funtion then returns the contribution rate of every link that is referenced (every link that is a value in the links rdd)
# The second line is to account for the contributions of links that are never referenced. (all keys that are in links but not contributions)
# The contribution of these links is set to zero while any link with a contribution stays the same.
# after this the rank is computed via the instructions for the assignment
for i in range(5):
    contributions = links.join(ranks).flatMap(lambda url: calc_contributions(url[1][0], url[1][1]))
    contributions = links.fullOuterJoin(contributions).mapValues(lambda url: assign_zero(url[1])).persist()
    ranks = contributions.reduceByKey(lambda x,y: x+y).mapValues(lambda rank: rank * .85 + .15).persist()

# Sorts by value then key of ranks 
ranks = ranks.sortBy(lambda x: (x[1], x[0]))
# Stores ranks as a directory of text files. 
ranks.saveAsTextFile(outfile)
print("----- %s seconds -----" % (time.time() - start_time))
