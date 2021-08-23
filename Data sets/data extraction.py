import pandas as pd

ATT=pd.read_csv("ATT.csv",encoding="utf-8",index_col=0)
HTL=pd.read_csv("HTL.csv",encoding="utf-8")
MOV=pd.read_csv("MOV.csv",encoding="utf-8")
PROD=pd.read_csv("PROD.csv",encoding="utf-8")
RES=pd.read_csv("RES.csv",encoding="utf-8")
RES1=pd.read_csv("RES1.csv",encoding="utf-8")
RES2=pd.read_csv("RES2.csv",encoding="utf-8")

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
dataset=pd.concat(objs=[ATT,HTL,MOV,PROD,RES,RES1,RES2])
print(dataset.head(),dataset.columns,dataset.shape)

df=pd.read_csv("df.csv")
print(df.isnull().sum())
print(df.head())
df.loc[df["rating"] == 1, "rating"] = -1
df.loc[df["rating"] == 2, "rating"] = -1
df.loc[df["rating"] == 3, "rating"] = 0
df.loc[df["rating"] == 4, "rating"] = 1
df.loc[df["rating"] == 5, "rating"] = 1
df=df[["review","rating"]]
df.columns=["text","polarity"]
print(df.head())
dataset=pd.concat(objs=[dataset,df])
print(dataset.head())
print(dataset.info)
print(dataset.describe())
dataset.sample().reset_index(drop=True)
print(dataset.head())
dataset.loc[dataset["polarity"]==1,"polarity"]=1
dataset.loc[dataset["polarity"]==0,"polarity"]=2
dataset.loc[dataset["polarity"]==-1,"polarity"]=3
dataset.to_csv("dataset.csv",index=False)