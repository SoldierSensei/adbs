from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator

sparkconf= SparkConf().setAppName('diabetes').setMaster('spark://master:7077')
sc = SparkContext.getOrCreate(conf=sparkconf)
sc.stop()
# create a SparkSession
spark = SparkSession.builder.master("spark://master:7077").appName('diabetes').getOrCreate()

# load the diabetes dataset as a DataFrame
df = spark.read.csv('hdfs://master:9000/diabetes.csv/', header=True, inferSchema=True)

# create a feature vector by combining the input columns
feature_cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
assembler = VectorAssembler(inputCols=feature_cols, outputCol='features')
data = assembler.transform(df).select('features', 'Outcome')

# split the data into training and testing sets
train, test = data.randomSplit([0.7, 0.3], seed=42)

# create a random forest classifier and train it on the training set
rf = RandomForestClassifier(labelCol='Outcome', featuresCol='features', numTrees=100, maxDepth=4, seed=42)
model = rf.fit(train)

# make predictions on the testing set
predictions = model.transform(test)

# evaluate the performance of the model using a binary classification evaluator
evaluator = BinaryClassificationEvaluator(labelCol='Outcome')
auc = evaluator.evaluate(predictions)
print(f'AUC: {auc}')

# save the trained model to Hadoop HDFS
model.save('hdfs://master:9000/RandModel')
