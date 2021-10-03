import pandas as pd

HTL=pd.read_csv("data sets/old/HTL.csv",encoding="utf-8")
MOV=pd.read_csv("data sets/old/MOV.csv",encoding="utf-8")
PROD=pd.read_csv("data sets/old/PROD.csv",encoding="utf-8")
RES=pd.read_csv("data sets/old/RES.csv",encoding="utf-8")
RES1=pd.read_csv("data sets/old/RES1.csv",encoding="utf-8")
RES2=pd.read_csv("data sets/old/RES2.csv",encoding="utf-8")

print(HTL.head(),HTL.columns)
print(MOV.head(),MOV.columns)
print(PROD.head(),PROD.columns)
print(RES.head(),RES.columns)
print(RES1.head(),RES1.columns)
print(RES2.head(),RES2.columns)
RES1.drop(columns=["user_id","restaurant_id"],inplace=True)
RES1=RES1[["text","polarity"]]
print(RES1.head(),RES1.columns)


HTL.loc[HTL["polarity"]==1,"polarity"]=1
HTL.loc[HTL["polarity"]==0,"polarity"]=2
HTL.loc[HTL["polarity"]==-1,"polarity"]=3
HTL.to_csv("data sets/new/hotels.csv",index=False)

MOV.loc[MOV["polarity"]==1,"polarity"]=1
MOV.loc[MOV["polarity"]==0,"polarity"]=2
MOV.loc[MOV["polarity"]==-1,"polarity"]=3
MOV.to_csv("data sets/new/movies.csv",index=False)

PROD.loc[PROD["polarity"]==1,"polarity"]=1
PROD.loc[PROD["polarity"]==0,"polarity"]=2
PROD.loc[PROD["polarity"]==-1,"polarity"]=3
PROD.to_csv("data sets/new/products.csv",index=False)

REST=pd.concat([RES,RES1,RES2])

REST.loc[REST["polarity"]==1,"polarity"]=1
REST.loc[REST["polarity"]==0,"polarity"]=2
REST.loc[REST["polarity"]==-1,"polarity"]=3
print(REST.head())
REST.to_csv("data sets/new/resturant.csv",index=False)