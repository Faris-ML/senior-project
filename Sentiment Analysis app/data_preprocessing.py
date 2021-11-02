import pandas as pd
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
import nltk
import re
import string
nltk.download('stopwords')

def transform_text(text):
    # first we define a list of arabic and english punctiations that we want to get rid of in our text

    punctuations = '''`÷×؛<>_()*&^%][ـ،/:"؟.,'{}~¦+|!”…“–ـ''' + string.punctuation

    # Arabic stop words with nltk
    stop_words = stopwords.words()

    # remove punctuations
    translator = str.maketrans('', '', punctuations)
    text = text.translate(translator)
    #text=re.sub(punctuations,'',text)


    text = ' '.join(word for word in text.split() if word not in stop_words)


    return text

def main(datapath:string,filenames:list,savepath:string):
    for i in filenames:
        df = pd.read_csv(datapath + "/" + i + ".csv")
        print("the data set info : \n", df.info, df.describe(), "\nthe data set shape : ", df.shape)
        print("null values check : ", "\n", df.isnull().sum())
        plt.hist(df["polarity"], color="r", bins=10)
        plt.title("reviews distribution")
        plt.xlabel("sentiments")
        plt.ylabel("count")
        plt.show()
        print("the number of negative reviews : ", df[df["polarity"] == 3]["polarity"].count())
        print("the number of natural reviews : ", df[df["polarity"] == 2]["polarity"].count())
        print("the number of positive reviews : ", df[df["polarity"] == 1]["polarity"].count())
        df['text'] = df['text'].apply(transform_text)
        print(df.head())
        df.to_csv(savepath+"/"+i+"_cleand.csv", index=False)

def excute():
    filepath = "data sets/new"
    file_names = ["movies", "hotels", "resturant"]
    main(datapath=filepath, filenames=file_names, savepath="data sets")



