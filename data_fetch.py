# import necessary packages 
import pandas as pd 
import psycopg2 
from sqlalchemy import create_engine 

# establish connection with the database 
conn = psycopg2.connect(host='127.0.0.1',port=5432,dbname='postgres',user='postgres',password='PasForNothing')
conn.autocommit=True
conn.commit()
# read the postgresql table 
table_df = pd.read_sql('''select * from "UTCL_Optimizer"."Model_Factors"''', conn)

# print the postgresql table loaded as 
# pandas dataframe 
print(table_df.shape) 
print(table_df.tail(5))
table_df.to_csv(r"C:/Users/deepak.pandey-v/Downloads/model_factor.csv", header=True, index_label=False,index=None)
print("Successfully csv generated")