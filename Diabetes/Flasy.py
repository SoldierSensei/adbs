from pyspark.ml.classification import RandomForestClassificationModel
from pyspark.ml.feature import VectorAssembler
import mysql.connector
from flask import Flask, render_template
import random

# create a Flask application
app = Flask(__name__)

# load the trained random forest model from Hadoop HDFS
model = RandomForestClassificationModel.load('hdfs://master:9000/RandModel')

# create a MySQL database connection
conn = mysql.connector.connect(
    host='localhost',
    user='your_username',
    password='your_password',
    database='your_database_name'
)

# create a cursor object to execute SQL queries
cursor = conn.cursor()

# create the patients table in MySQL if it doesn't already exist
cursor.execute("""
CREATE TABLE IF NOT EXISTS patients (
  Name VARCHAR(255),
  ID INT,
  Pregnancies INT,
  Glucose INT,
  BloodPressure INT,
  SkinThickness INT,
  Insulin INT,
  BMI FLOAT,
  DiabetesPedigreeFunction FLOAT,
  Age INT,
  Outcome INT
);
""")
conn.commit()

# define a function to make predictions using the trained model and save the results to the database
def predict(patient):
    # convert the patient details to a PySpark DataFrame
    patient_df = spark.createDataFrame([patient], ['Name', 'ID', 'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])

    # create a feature vector by combining the input columns
    feature_cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    assembler = VectorAssembler(inputCols=feature_cols, outputCol='features')
    data = assembler.transform(patient_df).select('features')

    # make a prediction using the trained model
    outcome = model.predict(data.first().features.toArray())

    # save the patient details and prediction outcome to the database
    cursor.execute('INSERT INTO patients (Name, ID, Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age, Outcome) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)', (*patient, outcome))
    conn.commit()

# define a route to display 10 random patients with their name and ID on a webpage
@app.route('/')
def index():
    # fetch 10 random patients from the database
    cursor.execute('SELECT Name, ID FROM patients ORDER BY RAND() LIMIT 10')
    patients = cursor.fetchall()

    # render the webpage with the patient details
    return render_template('index.html', patients=patients)

if __name__ == '__main__':
    # run the Flask application
    app.run(debug=True)
