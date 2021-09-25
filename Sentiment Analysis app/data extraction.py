import pandas as pd

ATT=pd.read_csv("data sets/ATT.csv",encoding="utf-8",index_col=0)
HTL=pd.read_csv("data sets/HTL.csv",encoding="utf-8")
MOV=pd.read_csv("data sets/MOV.csv",encoding="utf-8")
PROD=pd.read_csv("data sets/PROD.csv",encoding="utf-8")
RES=pd.read_csv("data sets/RES.csv",encoding="utf-8")
RES1=pd.read_csv("data sets/RES1.csv",encoding="utf-8")
RES2=pd.read_csv("data sets/RES2.csv",encoding="utf-8")

print(ATT.head(),ATT.columns)
print(HTL.head(),HTL.columns)
print(MOV.head(),MOV.columns)
print(PROD.head(),PROD.columns)
print(RES.head(),RES.columns)
print(RES1.head(),RES1.columns)
print(RES2.head(),RES2.columns)
RES1.drop(columns=["user_id","restaurant_id"],inplace=True)
RES1=RES1[["text","polarity"]]
print(RES1.head(),RES1.columns)

ATT.loc[ATT["polarity"]==1,"polarity"]=1
ATT.loc[ATT["polarity"]==0,"polarity"]=2
ATT.loc[ATT["polarity"]==-1,"polarity"]=3
ATT.to_csv("data sets/ATT_.csv",index=False)

HTL.loc[HTL["polarity"]==1,"polarity"]=1
HTL.loc[HTL["polarity"]==0,"polarity"]=2
HTL.loc[HTL["polarity"]==-1,"polarity"]=3
HTL.to_csv("data sets/HTL_.csv",index=False)

MOV.loc[MOV["polarity"]==1,"polarity"]=1
MOV.loc[MOV["polarity"]==0,"polarity"]=2
MOV.loc[MOV["polarity"]==-1,"polarity"]=3
MOV.to_csv("data sets/MOV_.csv",index=False)

PROD.loc[PROD["polarity"]==1,"polarity"]=1
PROD.loc[PROD["polarity"]==0,"polarity"]=2
PROD.loc[PROD["polarity"]==-1,"polarity"]=3
PROD.to_csv("data sets/PROD_.csv",index=False)

REST=pd.concat([RES,RES1,RES2])

REST.loc[REST["polarity"]==1,"polarity"]=1
REST.loc[REST["polarity"]==0,"polarity"]=2
REST.loc[REST["polarity"]==-1,"polarity"]=3
print(REST.head())
REST.to_csv("data sets/RES_.csv",index=False)