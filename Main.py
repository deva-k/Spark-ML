# Databricks notebook source
# MAGIC %md Setting Path

# COMMAND ----------

subscription_path = "/FileStore/tables/project/BDT2_1920_Subscriptions.csv"
complaints_path = "/FileStore/tables/project/BDT2_1920_Complaints.csv"
customers_path = "/FileStore/tables/project/BDT2_1920_Customers.csv"
delivery_path = "/FileStore/tables/project/BDT2_1920_Delivery.csv"
formula_path = "/FileStore/tables/project/BDT2_1920_Formula.csv"

# COMMAND ----------

# MAGIC %md Importing Functions

# COMMAND ----------

from pyspark.sql.functions import *
from functools import reduce
from operator import add

# COMMAND ----------

# MAGIC %sh pip install --upgrade pip

# COMMAND ----------

# MAGIC %sh pip install seaborn

# COMMAND ----------

# MAGIC %sh pip install --upgrade numpy

# COMMAND ----------

# import libraries
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T
from pyspark.sql import Window

from pyspark.ml import Pipeline, Model
from pyspark.ml import feature as FT
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
#from spark_stratifier import StratifiedCrossValidator
from pyspark.ml.evaluation import Evaluator, BinaryClassificationEvaluator
from pyspark.mllib.evaluation import BinaryClassificationMetrics
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.classification import LogisticRegression, DecisionTreeClassifier, RandomForestClassifier
from pyspark.ml.feature import OneHotEncoder, StringIndexer, IndexToString, VectorAssembler

import time
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# COMMAND ----------

# MAGIC %md Loading the Data

# COMMAND ----------

subscriptions=spark\
.read\
.format("csv")\
.option("header","true")\
.option("inferSchema","true")\
.load(subscription_path)

complaints=spark\
.read\
.format("csv")\
.option("header","true")\
.option("inferSchema","true")\
.load(complaints_path)

customers=spark\
.read\
.format("csv")\
.option("header","true")\
.option("inferSchema","true")\
.load(customers_path)

delivery=spark\
.read\
.format("csv")\
.option("header","true")\
.option("inferSchema","true")\
.load(delivery_path)

formula=spark\
.read\
.format("csv")\
.option("header","true")\
.option("inferSchema","true")\
.load(formula_path)

# COMMAND ----------

# MAGIC %md Getting labels

# COMMAND ----------

#get the last end date for each customer
label = subscriptions.groupby("CustomerID").agg(max("EndDate").alias("EndDate"))

# COMMAND ----------

#join the table to get other relevant information
label = label.join(subscriptions,["CustomerID","EndDate"],"left")

# COMMAND ----------

#drop duplicate values if any
label = label.dropDuplicates(subset = ["CustomerID"])

# COMMAND ----------

#select only required columns
label = label.select("CustomerID","StartDate","EndDate","RenewalDate",year("EndDate").alias("Year_EndDate"))

# COMMAND ----------

#condition for the churn label
label = label.withColumn("label", \
              when((label["RenewalDate"] == 'NA') & (label["Year_EndDate"] != 2019), 1).otherwise(0))

# COMMAND ----------

#selecting required columns
label = label.select("CustomerID","label")

# COMMAND ----------

label.groupby("label").count().show()

# COMMAND ----------

# MAGIC %md ##### Data Cleaning and preparation:

# COMMAND ----------

#Convert the type of the columns in the subscription table
s1 = subscriptions.withColumn("NbrMealsEXCEP",col("NbrMeals_EXCEP").cast("integer")).drop("NbrMeals_EXCEP")
s2 = s1.withColumn("GrossPrice_Formula",col("GrossFormulaPrice").cast("integer")).drop("GrossFormulaPrice")
s3 = s2.withColumn("NetPrice_Formula",col("NetFormulaPrice").cast("integer")).drop("NetFormulaPrice")
s4 = s3.withColumn("Price_per_meal",col("NbrMealsPrice").cast("double")).drop("NbrMealsPrice")
s5 = s4.withColumn("Discount_Product",col("ProductDiscount").cast("integer")).drop("ProductDiscount")
s6 = s5.withColumn("Discount_Formula",col("FormulaDiscount").cast("integer")).drop("FormulaDiscount")
s7 = s6.withColumn("Discount_Total",col("TotalDiscount").cast("integer")).drop("TotalDiscount")
s8 = s7.withColumn("Price_Total",col("TotalPrice").cast("integer")).drop("TotalPrice")
s9 = s8.withColumn("Credit_Total",col("TotalCredit").cast("integer")).drop("TotalCredit")
s10 = s9.withColumn("Date_Renewal",col("RenewalDate").cast("timestamp")).drop("RenewalDate")
s11 = s10.withColumn("Date_Payment",col("PaymentDate").cast("timestamp")).drop("PaymentDate")

subscriptions_converted = s11
#display(subscriptions_converted)

# COMMAND ----------

#check data types
subscriptions_converted.printSchema()

# COMMAND ----------

#create a dataframe
subscriptions_converted.createOrReplaceTempView("subscriptionsSQL")

# COMMAND ----------

subscriptions_new = subscriptions_converted
#subscriptions_new.select([count(when(isnan(c), c)).alias(c) for c in subscriptions_new.columns]).show()


# COMMAND ----------

# MAGIC %md Gathering some frequencies 

# COMMAND ----------

# DBTITLE 0,Frequency
#get year and customer id from the subscription table
sub_freq = subscriptions_new.select("CustomerID",year("StartDate").alias("year"))

# COMMAND ----------

#get the count of subscriptions per each year
sub_freq = sub_freq.groupby("CustomerID","year").count()

# COMMAND ----------

#transpose the year to columns
sub_freq = sub_freq.groupby('CustomerID').pivot("year").sum("count")

# COMMAND ----------

#replacing null valus with 0
sub_freq = sub_freq.na.fill(0)

# COMMAND ----------

#calculating total number of subscriptions
columns = ["2014","2015","2016","2017","2018","2019"]
sub_freq = sub_freq.withColumn("TotalSubscriptions" ,reduce(add, [col(x) for x in columns]))

# COMMAND ----------

#renaming the columns
sub_freq = sub_freq.select("CustomerID",col("2014").alias("Sub_2014"),col("2015").alias("Sub_2015"),col("2016").alias("Sub_2016") \
                           ,col("2017").alias("Sub_2017"),col("2018").alias("Sub_2018"),col("2019").alias("Sub_2019"),"TotalSubscriptions")

# COMMAND ----------

# MAGIC %md A) Subscriptions Table Cleaning and Preparation

# COMMAND ----------

#Replace the NA values in the numeric variables with 0:

subscriptions_new = subscriptions_new.withColumn("NbrMealsEXCEP", \
              when(subscriptions_new["NbrMealsEXCEP"] == 'NA', 0.).otherwise(subscriptions_new["NbrMealsEXCEP"]))
subscriptions_new = subscriptions_new.withColumn("NbrMeals_REG", \
              when(subscriptions_new["NbrMeals_REG"] == 'NA', 0.).otherwise(subscriptions_new["NbrMeals_REG"]))
subscriptions_new = subscriptions_new.withColumn("GrossPrice_Formula", \
              when(subscriptions_new["GrossPrice_Formula"] == 'NA', 0.).otherwise(subscriptions_new["GrossPrice_Formula"]))
subscriptions_new = subscriptions_new.withColumn("NetPrice_Formula", \
              when(subscriptions_new["NetPrice_Formula"] == 'NA', 0.).otherwise(subscriptions_new["NetPrice_Formula"]))
subscriptions_new = subscriptions_new.withColumn("Price_per_meal", \
              when(subscriptions_new["Price_per_meal"] == 'NA', 0.).otherwise(subscriptions_new["Price_per_meal"]))
subscriptions_new = subscriptions_new.withColumn("Discount_Product", \
              when(subscriptions_new["Discount_Product"] == 'NA', 0.).otherwise(subscriptions_new["Discount_Product"]))
subscriptions_new = subscriptions_new.withColumn("Discount_Total", \
              when(subscriptions_new["Discount_Total"] == 'NA', 0.).otherwise(subscriptions_new["Discount_Total"]))
subscriptions_new = subscriptions_new.withColumn("Discount_Formula", \
              when(subscriptions_new["Discount_Formula"] == 'NA', 0.).otherwise(subscriptions_new["Discount_Formula"]))
subscriptions_new = subscriptions_new.withColumn("Price_Total", \
              when(subscriptions_new["Price_Total"] == 'NA', 0.).otherwise(subscriptions_new["Price_Total"]))
subscriptions_new = subscriptions_new.withColumn("Credit_Total", \
              when(subscriptions_new["Credit_Total"] == 'NA', 0.).otherwise(subscriptions_new["Credit_Total"]))
subscriptions_new = subscriptions_new.withColumn("Price_per_meal", \
              when(subscriptions_new["Price_per_meal"] == 'NA', 0.).otherwise(subscriptions_new["Price_per_meal"]))

# COMMAND ----------

#To deal with the timestamp variables in the table, we will replace the missing values with the last EndDate (for missing values in Date_Renewal and Date_Payment):


subscriptions_new = subscriptions_new.withColumn("Date_Payment", \
               when(subscriptions_new["Date_Payment"] == 'NA', to_date(subscriptions_new["EndDate"],"yyyy/MM/dd")).otherwise(subscriptions_new["Date_Payment"]))


subscriptions_new = subscriptions_new.withColumn("Date_Renewal", \
               when(subscriptions_new["Date_Renewal"] == 'NA', to_date(subscriptions_new["EndDate"],"yyyy/MM/dd")).otherwise(subscriptions_new["Date_Renewal"]))

# COMMAND ----------

#Compute new variables for the numeric existing ones: min, max, avg

subscriptions_grouped = subscriptions_new.groupby('CustomerID').agg(round(avg("NbrMeals_REG"),2).alias("avg_NbrMeals_REG"),
                                                                                                round(max("NbrMeals_REG"),2).alias("max_NbrMeals_REG"),round(min("NbrMeals_REG"),2).alias("min_NbrMeals_REG"),
                                    round(avg("NbrMealsEXCEP"),2).alias("avg_NbrMealsEXCEP"),
                                                                    round(max("NbrMealsEXCEP"),2).alias("max_NbrMealsEXCEP"),
                                                                    round(min("NbrMealsEXCEP"),2).alias("min_NbrMealsEXCEP"),
                                                                    round(avg("GrossPrice_Formula"),2).alias("avg_GrossPrice_Formula"),
                                                                    round(max("GrossPrice_Formula"),2).alias("max_GrossPrice_Formula"),
                                                                    round(min("GrossPrice_Formula"),2).alias("min_GrossPrice_Formula"),
                                    round(avg("NetPrice_Formula"),2).alias("avg_NetPrice_Formula"),
                                                                    round(max("NetPrice_Formula"),2).alias("max_NetPrice_Formula"),
                                                                    round(min("NetPrice_Formula"),2).alias("min_NetPrice_Formula"),
                                    round(avg("Price_per_meal"),2).alias("avg_Price_per_meal"),
                                                                    round(max("Price_per_meal"),2).alias("max_Price_per_meal"),
                                                                    round(min("Price_per_meal"),2).alias("min_Price_per_meal"),
                                    round(avg("Discount_Product"),2).alias("avg_Discount_Product"),
                                                                    round(max("Discount_Product"),2).alias("max_Discount_Product"),
                                                                    round(min("Discount_Product"),2).alias("min_Discount_Product"),
                                    round(avg("Discount_Formula"),2).alias("avg_Discount_Formula"),
                                                                    round(max("Discount_Formula"),2).alias("max_Discount_Formula"),
                                                                    round(min("Discount_Formula"),2).alias("min_Discount_Formula"),
                                    round(avg("Discount_Total"),2).alias("avg_Discount_Total"),
                                                                    round(max("Discount_Total"),2).alias("max_Discount_Total"),
                                                                    round(min("Discount_Total"),2).alias("min_Discount_Total"),
                                    round(avg("Price_Total"),2).alias("avg_Price_Total"),
                                                                    round(max("Price_Total"),2).alias("max_Price_Total"),
                                                                    round(min("Price_Total"),2).alias("min_Price_Total"),                       count("Price_Total").alias("count_Price_Total"),
                                round(avg("Credit_Total"),2).alias("avg_Credit_Total"),
                                                                    max("Credit_Total").alias("max_Credit_Total"),
                                                                    min("Credit_Total").alias("min_Credit_Total"),
                                                                    count("Credit_Total").alias("count_Credit_Total")
                                    )

# COMMAND ----------

#We need to create a new table with the dates separately before we can group the subscriptions table by CustomerID
subscriptions_date = subscriptions_new.select("SubscriptionID","CustomerID","StartDate","EndDate","Date_Payment","Date_Renewal")
#subscriptions_date = subscriptions_date.groupby('CustomerID').agg(min("StartDate"),max("EndDate"),max("Date_Payment"),max("Date_Renewal"))
#display(subscriptions_date)

# COMMAND ----------

#Next, we calculate the time difference between the last end date and the first start date; and the number of days between the last payment date and the last renewat
#and the number of days since February 1st and the last renewal date.
subscriptions_date = subscriptions_date.withColumn("AggSubscriptionDuration", 
              datediff(to_date("EndDate","yyyy/MM/dd"),
                       to_date("StartDate","yyyy/MM/dd")))\
                                    .withColumn("NbDayLastPayment", 
              datediff(to_date(lit("2019-02-01")),
                       to_date("Date_Payment")))\
                                    .withColumn("NbDayLastRenewal", 
              datediff(to_date(lit("2019-02-01")),
                       to_date("Date_Renewal")))

#display(subscriptions_date)

# COMMAND ----------

#getting aggregations for dates
subscriptions_date = subscriptions_date.groupby('CustomerID').agg(sum("AggSubscriptionDuration").alias("AggSubscriptionDuration"),min("NbDayLastPayment").alias("NbDayLastPayment"),min("NbDayLastRenewal").alias("NbDayLastRenewal"))

# COMMAND ----------

# MAGIC %md Next step: dummy encoding for the variables

# COMMAND ----------

import pyspark.sql.functions as F


# COMMAND ----------

#selecting required columns
subscriptions_categ = subscriptions_new.select("CustomerID","ProductName")

# COMMAND ----------

#Dummy encoding for the Subscriptions_converted table

categ = subscriptions_categ.select('ProductName').distinct().rdd.flatMap(lambda x:x).collect()
exprs = [F.when(F.col('ProductName') == cat,1).otherwise(0)\
            .alias(str(cat)) for cat in categ]
subscriptions_categ = subscriptions_categ.select(exprs+subscriptions_categ.columns)

# COMMAND ----------

subscriptions_categ =subscriptions_categ.withColumnRenamed('Grub Flexi (excl. staff)', "Grub Flexi")
subscriptions_categ =subscriptions_categ.withColumnRenamed('Grub Maxi (incl. staff)', "Grub Maxi")

# COMMAND ----------

subscriptions_categ = subscriptions_categ.drop("ProductName")

# COMMAND ----------

subscriptions_categ = subscriptions_categ.groupby('CustomerID').agg(sum(col('Custom Events')).alias("TotalCustomEvents"),sum(col('Grub Flexi')).alias("TotalGrubFlexi"),sum(col('Grub Mini')).alias("TotalGrubMini"),\
                                                                        sum(col('Grub Maxi')).alias("TotalGrubMaxi"))

# COMMAND ----------

#joining all the df's generated through subscription
subscriptions_1 = subscriptions_categ.join(subscriptions_grouped, ["CustomerID"])
#display(subscriptions_1)

# COMMAND ----------

subscriptions_2 = subscriptions_1.join(subscriptions_date, ["CustomerID"])
#display(subscriptions_2)

# COMMAND ----------

subscriptions_cleaned = subscriptions_2

# COMMAND ----------

# MAGIC %md B) Complaints Table Cleaning and Preparation

# COMMAND ----------

#Replacing NA Values

complaints = complaints.withColumn("FeedbackTypeDesc", \
              when(complaints["FeedbackTypeDesc"] == "NA", "No_Feedback").otherwise(complaints["FeedbackTypeDesc"]))

complaints = complaints.withColumn("ProductName", \
              when(complaints["ProductName"] == "NA", "Other_Product").otherwise(complaints["ProductName"]))

complaints = complaints.withColumn("SolutionTypeDesc", \
              F.when(complaints["SolutionTypeDesc"] == "NA", "Solution_Other").otherwise(complaints["SolutionTypeDesc"]))

complaints = complaints.withColumn("SolutionTypeDesc", \
              F.when(complaints["SolutionTypeDesc"] == "other", "Solution_Other").otherwise(complaints["SolutionTypeDesc"]))

complaints = complaints.withColumn("ComplaintTypeDesc", \
               F.when(complaints["ComplaintTypeDesc"] == "other", "Diff_Complaint").otherwise(complaints["ComplaintTypeDesc"]))

complaints = complaints.withColumn("ComplaintTypeDesc", \
              F.when(complaints["ComplaintTypeDesc"] == "NA", "Other_Complaint").otherwise(complaints["ComplaintTypeDesc"]))



# COMMAND ----------

#Dummy Encoding and variable drops
import pyspark.sql.functions as F 
categ = complaints.select('FeedbackTypeDesc').distinct().rdd.flatMap(lambda x:x).collect()
exprs = [F.when(F.col('FeedbackTypeDesc') == cat,1).otherwise(0)\
            .alias(str(cat)) for cat in categ]
complaints = complaints.select(exprs+complaints.columns)
feedDrop = ["FeedbackTypeDesc", "FeedbackTypeID"]
complaints = complaints.drop(*feedDrop)

# COMMAND ----------

categ = complaints.select('ProductName').distinct().rdd.flatMap(lambda x:x).collect()
exprs = [F.when(F.col('ProductName') == cat,1).otherwise(0)\
            .alias(str(cat)) for cat in categ]
complaints = complaints.select(exprs+complaints.columns)

#Rename columns
complaints = complaints.withColumnRenamed('Grub Flexi (excl. staff)', "Grub Flexi")
complaints = complaints.withColumnRenamed('Grub Maxi (incl. staff)', "Grub Maxi")


# COMMAND ----------

categ = complaints.select('SolutionTypeDesc').distinct().rdd.flatMap(lambda x:x).collect()
exprs = [F.when(F.col('SolutionTypeDesc') == cat,1).otherwise(0)\
            .alias(str(cat)) for cat in categ]
complaints = complaints.select(exprs+complaints.columns)


FeedDrop = ["SolutionTypeDesc", "SolutionTypeID"]
complaints = complaints.drop(*FeedDrop)

# COMMAND ----------


categ = complaints.select('ComplaintTypeDesc').distinct().rdd.flatMap(lambda x:x).collect()
exprs = [F.when(F.col('ComplaintTypeDesc') == cat,1).otherwise(0)\
            .alias(str(cat)) for cat in categ]
complaints = complaints.select(exprs+complaints.columns)

FeedDrop = ["ComplaintTypeDesc", "ComplaintTypeID"]
complaints = complaints.drop(*FeedDrop)

# COMMAND ----------

# MAGIC %md Create a dataframe with the dates:

# COMMAND ----------

complaint_date = complaints.select("CustomerID","ComplaintDate")
#display(complaint_date)

# COMMAND ----------

complaint_date = complaint_date.groupby("CustomerID").agg(F.max("ComplaintDate"))

# COMMAND ----------

#See the number of days since the last complaint and drop the last complaint date column
complaint_date = complaint_date.withColumn("NbDayLastComplaint", 
              datediff(to_date(lit("2019-02-01")),
                       to_date("max(ComplaintDate)","yyyy/MM/dd")))
complaint_date = complaint_date.drop("max(ComplaintDate)")
complaint_date.count() #We have the right number of observations

# COMMAND ----------

#drop redundant or no longer useful variables in the complaints table
FeedDrop = ["ComplaintID", "ComplaintDate","ProductID","ProductName"]
complaints = complaints.drop(*FeedDrop)

# COMMAND ----------

complaints = complaints.groupby("CustomerID").sum()

# COMMAND ----------

complaints= complaints.drop("sum(CustomerID)")

# COMMAND ----------

#Rename columns
complaints = complaints.withColumnRenamed('sum(employees were rude)', "Complaint_Employees_Rude")
complaints = complaints.withColumnRenamed('sum(billing/invoice not correct)', "Complaint_Billing_Problem")
complaints = complaints.withColumnRenamed('sum(food quantity was insufficient)', "Complaint_Quantity")
complaints = complaints.withColumnRenamed('sum(food quality not good)', "Complaint_Quality")
complaints = complaints.withColumnRenamed('sum(food was cold)', "Complaint_Food_Cold")
complaints = complaints.withColumnRenamed('sum(order not correct)', "Complaint_Incorrect_Order")
complaints = complaints.withColumnRenamed('sum(Diff_Complaint)', "Complaint_Other")
complaints = complaints.withColumnRenamed('sum(late delivery)', "Complaint_Late_Delivery")
complaints = complaints.withColumnRenamed('sum(poor hygiene)', "Complaint_Poor_Hygiene")
complaints = complaints.withColumnRenamed('sum(exceptional price discount)', "Sol_Excep_Discount")
complaints = complaints.withColumnRenamed('sum(free additional meal service)', "Sol_Free_Service")
complaints = complaints.withColumnRenamed('sum(no compensation)', "Sol_No_Compensation")
complaints = complaints.withColumnRenamed('sum(Solution_Other)', "Sol_Other")
complaints = complaints.withColumnRenamed('sum(Other_Product)', "Other_Product")
complaints = complaints.withColumnRenamed('sum(Custom Events)', "Customer_Events")
complaints = complaints.withColumnRenamed('sum(Grub Flexi)', "Grub Flexi")
complaints = complaints.withColumnRenamed('sum(Grub Mini)', "Grub Mini")
complaints = complaints.withColumnRenamed('sum(Grub Maxi)', "Grub Maxi")
complaints = complaints.withColumnRenamed('sum(No_Feedback)', "No_Feedback")
complaints = complaints.withColumnRenamed('sum(not satisfied)', "Not_Satisfied")
complaints = complaints.withColumnRenamed('sum(no response)', "No_Response")
complaints = complaints.withColumnRenamed('sum(other)', "Other_Feedback")
complaints = complaints.withColumnRenamed('sum(satisfied)', "Satisfied")


# COMMAND ----------

#total number of complaints
columns = ["Complaint_Employees_Rude","Complaint_Billing_Problem","Complaint_Quantity","Complaint_Quality","Complaint_Food_Cold","Complaint_Incorrect_Order","Complaint_Other","Complaint_Late_Delivery","Complaint_Poor_Hygiene","Sol_Excep_Discount","Sol_Free_Service","Sol_No_Compensation","Sol_Other","Other_Product","Customer_Events","Grub Flexi","Grub Mini","Grub Maxi","No_Feedback","Not_Satisfied","No_Response","Other_Feedback","Satisfied"]
complaints = complaints.withColumn("NbrTotalComplaints" ,reduce(add, [col(x) for x in columns]))

# COMMAND ----------

#complaints_cleaned = complaints.join(complaints_number, ["CustomerID"])

# COMMAND ----------

#Final step: join complaints and complaint_date tables:
complaints_cleaned = complaints.join(complaint_date, ["CustomerID"])

# COMMAND ----------

display(complaints_cleaned)

# COMMAND ----------

# MAGIC %md C) Customers Table Cleaning and Preparation

# COMMAND ----------

#select regions from customers
customers = customers.select("CustomerID","Region")

# COMMAND ----------

customers = customers.withColumn("Region_Cust",concat(lit("Region"),"Region"))

# COMMAND ----------

#customers_cleaned
customers_cleaned = customers.drop("Region")

# COMMAND ----------

# MAGIC %md D) Delivery Table Cleaning and Preparation

# COMMAND ----------

#Replacing NA Values

delivery = delivery.withColumn("DeliveryClass", \
              when(delivery["DeliveryClass"] == "null", "Not_Specified").otherwise(delivery["DeliveryClass"]))

# COMMAND ----------

sub_del = subscriptions.select("CustomerID","SubscriptionID")

# COMMAND ----------

delivery = delivery.join(sub_del,["SubscriptionID"],"left")

# COMMAND ----------

#Dummy Encoding 
categ = delivery.select('DeliveryClass').distinct().rdd.flatMap(lambda x:x).collect()
exprs = [when(col('DeliveryClass') == cat,1).otherwise(0)\
            .alias(str(cat)) for cat in categ]
delivery = delivery.select(exprs+delivery.columns)

# COMMAND ----------

categ = delivery.select('DeliveryTypeName').distinct().rdd.flatMap(lambda x:x).collect()
exprs = [when(col('DeliveryTypeName') == cat,1).otherwise(0)\
            .alias(str(cat)) for cat in categ]
delivery = delivery.select(exprs+delivery.columns)

# COMMAND ----------

FeedDrop = ["DeliveryClass", "DeliveryTypeName","DeliveryID"]
delivery = delivery.drop(*FeedDrop)


# COMMAND ----------

#Creating a Delivery Date DF

delivery_date=delivery.select("CustomerID", "SubscriptionID", "DeliveryDate")

# COMMAND ----------

delivery_date = delivery_date.groupby('CustomerID').agg(F.max("DeliveryDate"))

# COMMAND ----------

delivery_date = delivery_date.withColumn("NbDayLastDelivery", 
              datediff(to_date(lit("2019-02-01")),
                       to_date("max(DeliveryDate)","yyyy/MM/dd")))

# COMMAND ----------

delivery = delivery.groupby("CustomerID").sum()

# COMMAND ----------

delivery_clean = delivery.join(delivery_date, ["CustomerID"])

# COMMAND ----------

drop_cols = ["sum(SubscriptionID)","sum(CustomerID)","max(DeliveryDate)"]
delivery_cleaned = delivery_clean.drop(*drop_cols)

# COMMAND ----------

#rename Columns
delivery_cleaned = delivery_cleaned.withColumnRenamed('sum(delivery w/ staff)', "Delivery_With_Staff")\
                  .withColumnRenamed('sum(drop off))', "DropOff_Delivery")\
                  .withColumnRenamed('sum(delivery w/o staff)','Delivery_Without_Staff')\
                  .withColumnRenamed('sum(None)',"DelType_None")\
                  .withColumnRenamed('sum(NOR)',"Normal_Delivery")\
                  .withColumnRenamed('sum(ABN)',"Abnormal_Delivery")


# COMMAND ----------

# MAGIC %md E) Formula Table Cleaning and Preparation

# COMMAND ----------

sub_form = subscriptions.select("CustomerID","FormulaID")

# COMMAND ----------

#getting the fomula names and durations
joinType="left"
joinExpression = ["FormulaID"]
sub_form = sub_form.join(formula,joinExpression,joinType)

# COMMAND ----------

sub_form = sub_form.groupby("CustomerID","FormulaType").count()

# COMMAND ----------

formula_cleaned = sub_form.groupby("CustomerID").pivot("FormulaType").sum("count")

# COMMAND ----------

formula_cleaned = formula_cleaned.na.fill(0)

# COMMAND ----------

# MAGIC %md ##### Joining all tables together 

# COMMAND ----------

#subscriptions_cleaned
#delivery_cleaned
#customers_cleaned
#complaints_cleaned
#formula_cleaned
#sub_freq

# COMMAND ----------

joinType="left"
joinExpression = ["CustomerID"]
main = subscriptions_cleaned.join(customers_cleaned,joinExpression,joinType)

# COMMAND ----------

joinType="left"
joinExpression = ["CustomerID"]
main = main.join(label,joinExpression,joinType)

# COMMAND ----------

joinType="left"
joinExpression = ["CustomerID"]
main = main.join(complaints_cleaned,joinExpression,joinType)

# COMMAND ----------

joinType="left"
joinExpression = ["CustomerID"]
main = main.join(delivery_cleaned,joinExpression,joinType)

# COMMAND ----------

joinType="left"
joinExpression = ["CustomerID"]
main = main.join(sub_freq,joinExpression,joinType)

# COMMAND ----------

joinType="left"
joinExpression = ["CustomerID"]
main = main.join(formula_cleaned,joinExpression,joinType)

# COMMAND ----------

#replacing all null values
main = main.na.fill(0)

# COMMAND ----------

main.count()

# COMMAND ----------

display(main)

# COMMAND ----------

# MAGIC %md ### MODELING

# COMMAND ----------

# convert to pandas dataframe for easy visualization
user_pd = main.toPandas()

plt.figure(figsize=(6, 5))
sns.countplot(x='label', data=user_pd)
plt.savefig('dist_churn.png')
display(plt.show())

# COMMAND ----------

# prepare training and test data
train, test = main.randomSplit([0.7, 0.3],seed=12)

# COMMAND ----------

#Ration of churners to non-churners in train and test sets
print('train set:')
train.groupBy('label').count().show()
print('test set:')
test.groupBy('label').count().show()

# COMMAND ----------

# MAGIC %md Feature selection through correlation

# COMMAND ----------

#converting train to pandas dataframe to get column and work on correlation
ctrain = train.toPandas()
dtrain = ctrain
ctrain = ctrain.drop(["CustomerID","label","DelType_None"], axis=1)
corr = ctrain.corr()

# COMMAND ----------

#dropping categorical variable from the correlation 
ctrain = ctrain.drop("Region_Cust", axis=1)

# COMMAND ----------

#selecting the columns based on the correlation
columns = np.full((corr.shape[0],), True, dtype=bool)
for i in range(corr.shape[0]):
    for j in range(i+1, corr.shape[0]):
        if corr.iloc[i,j] >= 0.9:
            if columns[j]:
                columns[j] = False
selected_columns = ctrain.columns[columns]
ctrain = ctrain[selected_columns]

# COMMAND ----------

#build regression model and calculate p values
#selected_columns = selected_columns[1:].values
import statsmodels.formula.api as sm
def backwardElimination(x, Y, sl, columns):
    numVars = len(x[0])
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(Y, x).fit()
        maxVar = np.max(regressor_OLS.pvalues).astype(float)
        if maxVar > sl:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    x = np.delete(x, j, 1)
                    columns = np.delete(columns, j)
                    
    regressor_OLS.summary()
    return x, columns
SL = 0.05
data_modeled, selected_columns = backwardElimination(ctrain.iloc[:,:].values, dtrain["label"].values, SL, selected_columns)

# COMMAND ----------

selected_columns

# COMMAND ----------

#the final list of variables to be used
numerical_cols = selected_columns.tolist()
categorical_cols = [item[0] for item in main.dtypes if item[1].startswith('string')] 

# COMMAND ----------

# MAGIC %md Exploratory Data Analysis

# COMMAND ----------

plt.figure(figsize=(12,16))

for i in range(len(numerical_cols)):
    plt.subplot(8,3,i+1)
    plt.tight_layout()
    sns.distplot(user_pd[user_pd['label']==0][numerical_cols[i]],
                 hist=False, norm_hist=True, kde_kws={'shade': True, 'linewidth': 2})
    sns.distplot(user_pd[user_pd['label']==1][numerical_cols[i]],
                 hist=False, norm_hist =True, kde_kws={'shade': True, 'linewidth': 2})
    plt.legend(['Not Churned','Churned'])
    plt.title(numerical_cols[i])
    plt.xlabel(' ')
    plt.yticks([])

plt.savefig('dist_numerical.png')
display(plt.show())

# COMMAND ----------

# a funciton to plot correlations among columns
def plot_corr(cols, figsize=(10,10), filename=None, df=user_pd):
    plt.figure(figsize=figsize)
    sns.heatmap(df[cols].corr(),
                square=True, cmap='YlGnBu', annot=True,
                vmin=-1, vmax=1)
    plt.ylim(len(cols), 0)
    if filename:
        plt.savefig(filename)
    display(plt.show())

# COMMAND ----------

# correlations between numerical features
user_pd = main.toPandas()
user_pd.shape
numerical_cols = numerical_cols
plot_corr(numerical_cols, figsize=(20,20), filename='corr_full.png')


# COMMAND ----------

  # build data-process stages to encode, scale and assemble features
stages = []

# encode categorical features
for col in categorical_cols:
    indexer = FT.StringIndexer(inputCol=col, outputCol=col+'_idx')
    indexer = indexer.setHandleInvalid("skip") 
    encoder = FT.OneHotEncoderEstimator(inputCols=[indexer.getOutputCol()], outputCols=[col+'_vec'])
    stages += [indexer, encoder]

# scale numeric features
for col in numerical_cols: 
    assembler = FT.VectorAssembler(inputCols=[col], outputCol=col+'_vec')
    scaler = FT.StandardScaler(inputCol=col+'_vec', outputCol=col+'_scl')
    stages += [assembler, scaler]

# assemble features
input_cols = [c+'_vec' for c in categorical_cols] + [c+'_scl' for c in numerical_cols]
assembler = FT.VectorAssembler(inputCols=input_cols, outputCol='features')
stages += [assembler]

# COMMAND ----------

#Fitting
def fit_predict(pipeline):
    # fit on train set
    start = time.time()
    model = pipeline.fit(train)
    end = time.time()
    print(f'train time: {end-start:.0f}s')

    # predict on test set
    pred_train = model.transform(train)
    pred_test = model.transform(test)

    return model, pred_train, pred_test

# COMMAND ----------

#Evaluation Metrics
def evaluate(pred, beta=1):
    # true positive
    tp = pred.filter((pred.prediction==1)&(pred.label==1)).count()
    # false positive
    fp = pred.filter((pred.prediction==1)&(pred.label==0)).count()
    # false negative
    fn = pred.filter((pred.prediction==0)&(pred.label==1)).count()
    # true negative
    tn = pred.filter((pred.prediction==0)&(pred.label==0)).count()

    # calculate scores
    accuracy = (tp+tn)/(tn+fp+fn+tp)
    precision = tp/(tp+fp) if (tp+fp)!=0 else 0
    recall = tp/(tp+fn) if (tp+fn)!=0 else 0
    f_beta = (1+beta**2)*(precision*recall)/(beta**2*precision+recall) if (precision+recall)!=0 else 0
    
    #AUC
    preds = pred.select("prediction","label")
    out = preds.rdd.map(lambda x: (float(x[0]), float(x[1])))
    metrics = BinaryClassificationMetrics(out)
    AUC = metrics.areaUnderROC


    print(f'f{beta}-score:{f_beta:.2f}', )
    print(f'precision:{precision:.2f}')
    print(f'recall:{recall:.2f}')
    print(f'accuracy:{accuracy:.2f}')
    print('confusion matrix:')
    print(f'TP:{tp:.1f}\t | FP:{fp:.1f}')
    print(f'FN:{fn:.1f}\t | TN:{tn:.1f}')
    print(f'AUC : {AUC}')
    

# COMMAND ----------

#Run the Algorithms
lr = LogisticRegression(maxIter=100)
dtc = DecisionTreeClassifier(seed=5)
rfc = RandomForestClassifier(seed=12)

pipelines = [
    Pipeline(stages=stages+[lr]),
    Pipeline(stages=stages+[dtc]),
    Pipeline(stages=stages+[rfc])
]

for model, pipeline in zip([lr,dtc,rfc], pipelines):
    print('\n', type(model))
    model, pred_train, pred_test = fit_predict(pipeline)
    print('{:-^30}'.format('pred_train'))
    evaluate(pred_train)
    print('{:-^30}'.format('pred_test'))
    evaluate(pred_test)

# COMMAND ----------

#HyperParameter tuning and cross validation

pipeline = Pipeline(stages=stages+[rfc])
param_grid = (ParamGridBuilder().addGrid(rfc.numTrees,[50,75,100,125]).build())
evaluator = BinaryClassificationEvaluator()

cv = CrossValidator(estimator=pipeline, estimatorParamMaps=param_grid,
                     evaluator=evaluator, numFolds=3)


# COMMAND ----------

# fit, predict and evaluate the cross validation model
cv_model, cv_train, cv_test = fit_predict(cv)
print('{:-^30}'.format('pred_train'))
evaluate(cv_train)
print('{:-^30}'.format('pred_test'))
evaluate(cv_test)


# COMMAND ----------

#best parameters
cvBestPipeline = cv_model.bestModel
cvBestLRModel = cvBestPipeline.stages[-1]._java_obj.parent() #the stages function refers to the stage in the pipelinemodelA
print("Best DT model:")
print("** NumTrees: " ,(cvBestLRModel.getNumTrees()))

# COMMAND ----------

#get predictions and probabilities
preds_test = cv_model.transform(test)\
  .select("CustomerID","prediction", "label","probability")

# COMMAND ----------

display(preds_test)

# COMMAND ----------

#manipulating the probability vector to get probability of churn
from pyspark.sql.functions import udf
from pyspark.sql.types import FloatType

secondelement=udf(lambda v:float(v[1]),FloatType())
test_proba = preds_test.select("CustomerID",secondelement('probability').alias("Proba_churn"))

# COMMAND ----------

test_proba.count()

# COMMAND ----------

#getting predictions and probabilities from the train set customers
preds_train = cv_model.transform(train)\
  .select("CustomerID","prediction", "label","probability")

# COMMAND ----------

#manipulating the probability vector to get probability of churn
secondelement=udf(lambda v:float(v[1]),FloatType())
train_proba = preds_train.select("CustomerID",secondelement('probability').alias("Proba_churn"))

# COMMAND ----------

train_proba.count()

# COMMAND ----------

#stacking both the probability dataframes
from functools import reduce  # For Python 3.x
from pyspark.sql import DataFrame

def unionAll(*df):
    return reduce(DataFrame.unionAll, df)

probabilities = unionAll(test_proba,train_proba)

# COMMAND ----------

display(probabilities)

# COMMAND ----------

probabilities.count()

# COMMAND ----------

joinType="left"
joinExpression = ["CustomerID"]
main_probabilities = main.join(probabilities,joinExpression,joinType)
