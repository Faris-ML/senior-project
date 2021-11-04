from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import preprocessing
import pandas as pd
import pickle as pkl

def normalize(data_frame:pd.DataFrame):
    min_max = preprocessing.MaxAbsScaler() # normalize data using min max scaler
    df_min_max = min_max.fit_transform(data_frame)

    return df_min_max


#__________     train hotels data   _____________
df=pd.read_csv("data sets/new/hotels.csv",encoding="utf-8")
text=df["text"].astype("U").to_numpy().ravel()

#split lables and data
Y=df["polarity"]
X=df["text"].astype("U").to_numpy()

#define the model
model=LogisticRegression(solver="saga",max_iter=100000,multi_class="ovr")
#defain BOW algorithm
transformer=CountVectorizer(ngram_range=(1,3)).fit(text)
#normlize
x_train = normalize(transformer.transform(X.ravel()))
#fit the model
model.fit(x_train, Y)
#save the model
TransformerName="HTL_BOW.sav"
pkl.dump(transformer, open(TransformerName, 'wb'))
ModelName="HTL_model.sav"
pkl.dump(model, open(ModelName, 'wb'))

#____________________      train resturant data  _______________

df=pd.read_csv("data sets/new/resturant.csv",encoding="utf-8")
text=df["text"].astype("U").to_numpy().ravel()

#split lables and data
Y=df["polarity"]
X=df["text"].astype("U").to_numpy()

#define the model
model=LogisticRegression(solver="saga",max_iter=100000,multi_class="ovr")

#defain BOW algorithm
transformer=TfidfVectorizer(ngram_range=(1,3)).fit(text)

#normlize
x_train = normalize(transformer.transform(X.ravel()))

#fit the model
model.fit(x_train, Y)

#save the model
TransformerName="RES_BOW.sav"
pkl.dump(transformer, open(TransformerName, 'wb'))
ModelName="RES_model.sav"
pkl.dump(model, open(ModelName, 'wb'))

# _________________ train on movies data ______________

df=pd.read_csv("data sets/new/movies.csv",encoding="utf-8")
text=df["text"].astype("U").to_numpy().ravel()

#split lables and data
Y=df["polarity"]
X=df["text"].astype("U").to_numpy()

#define the model
model=LogisticRegression(solver="saga",max_iter=100000,multi_class="ovr")

#defain BOW algorithm
transformer=CountVectorizer(ngram_range=(1,3)).fit(text)

#normlize
x_train = normalize(transformer.transform(X.ravel()))

#fit the model
model.fit(x_train, Y)

#save the model
TransformerName="MOV_BOW.sav"
pkl.dump(transformer, open(TransformerName, 'wb'))
ModelName="MOV_model.sav"
pkl.dump(model, open(ModelName, 'wb'))



