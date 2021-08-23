import sklearn as sk
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import preprocessing
import numpy as np
import pandas as pd

def downsample(df:pd.DataFrame, label_col_name:str) -> pd.DataFrame:
    # find the number of observations in the smallest group
    nmin = df[label_col_name].value_counts().min()
    return (df
            # split the dataframe per group
            .groupby(label_col_name)
            # sample nmin observations from each group
            .apply(lambda x: x.sample(nmin))
            # recombine the dataframes
            .reset_index(drop=True)
            )

def normalize(data_frame):
    min_max = preprocessing.MaxAbsScaler() # normalize data using min max scaler
    df_min_max = min_max.fit_transform(data_frame)

    return df_min_max

#initialize the models algorithms
NBmodel=MultinomialNB()
LRmodel=LogisticRegression(solver="saga",max_iter=10000,multi_class="ovr",C=100)
SVMmodel=sk.svm.SVC(decision_function_shape="ovr",max_iter=10000,C=100)
RFmodel=RandomForestClassifier(n_estimators=20,n_jobs=-1)

dicm ={1:"naive bayas",2:"logistic regression",3:"SVM",4:"random forest"}

#make list of models to loop over it
models=[NBmodel,LRmodel,SVMmodel,RFmodel]

#load the text
df=pd.read_csv("dataset_cleand.csv",encoding="utf-8")
print(df.head())
print(df["polarity"].value_counts())
text=df["text"].astype("U").to_numpy()

#under sampling the data
df=downsample(df,"polarity")
print(df["polarity"].value_counts())

#split lables and data
Y=df["polarity"]
X=df["text"].astype("U").to_numpy()

#split to train and test set
X_train,X_test,y_train,y_test=sk.model_selection.train_test_split(X,Y,test_size=0.08)
#make a list of n-gram text to number algorithm  to loop over it
CV=[CountVectorizer(ngram_range=(1,1)).fit(text),CountVectorizer(ngram_range=(1,2)).fit(text),CountVectorizer(ngram_range=(1,3)).fit(text)]
TF_IDF=[TfidfVectorizer(ngram_range=(1,1)).fit(text),TfidfVectorizer(ngram_range=(1,2)).fit(text),TfidfVectorizer(ngram_range=(1,3).fit(text))]

#loop over models and text to number algorithims and pick the best score
scores=np.zeros([4,2,3])
for i in range(0,len(CV)):
    x_train= normalize(CV[i].transform(X_train))
    x_test= normalize(CV[i].transform(X_test))
    for j in range(0,len(models)):
        models[j].fit(x_train,y_train)
        y_pred=models[j].predict(x_test)
        y_pred2=models[j].predict(x_train)
        scores[j,0,i]=sk.metrics.accuracy_score(y_true=y_test,y_pred=y_pred)
        train_score=sk.metrics.accuracy_score(y_true=y_train,y_pred=y_pred2)
        print("the model "+dicm[j+1]+" and countvectorizer "+str(i+1)+" gram accuray { train :"+str(train_score)+"-- test :"+str(scores[j,0,i])+" }")

for k in range(0,len(TF_IDF)):
    x_train = normalize(TF_IDF[i].transform(X_train))
    x_test = normalize(TF_IDF[i].transform(X_test))
    for j in range(0,len(models)):
        models[j].fit(x_train, y_train)
        y_pred = normalize(models[j].predict(x_test))
        y_pred2 = normalize(models[j].predict(x_train))
        scores[j, 1, i] = sk.metrics.accuracy_score(y_true=y_test, y_pred=y_pred)
        train_score = sk.metrics.accuracy_score(y_true=y_train, y_pred=y_pred2)
        print("the model "+dicm[j+1]+" and TF-IDF "+str(i+1)+" gram accuray { train :"+str(train_score)+"-- test :"+str(scores[j,1,i])+" }")
print(scores)






