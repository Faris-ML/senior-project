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

    arabic_diacritics = re.compile("""
                                 ّ    | # Shadda
                                 َ    | # Fatha
                                 ً    | # Tanwin Fath
                                 ُ    | # Damma
                                 ٌ    | # Tanwin Damm
                                 ِ    | # Kasra
                                 ٍ    | # Tanwin Kasr
                                 ْ    | # Sukun
                                 ـ     # Tatwil/Kashida
                             """, re.VERBOSE)

    # remove punctuations
    translator = str.maketrans('', '', punctuations)
    text = text.translate(translator)
    #text=re.sub(punctuations,'',text)
    # remove Tashkeel
    text = re.sub(arabic_diacritics, '', text)

    # remove longation
    text = re.sub("[إأآا]", "ا", text)
    text = re.sub("ى", "ي", text)
    text = re.sub("ؤ", "ء", text)
    text = re.sub("ئ", "ء", text)
    text = re.sub("ة", "ه", text)
    text = re.sub("گ", "ك", text)

    text = ' '.join(word for word in text.split() if word not in stop_words)


    return text

df=pd.read_csv("dataset.csv")
print("the data set info : \n",df.info,df.describe(),"\nthe data set shape : ",df.shape)
print("null values check : ","\n",df.isnull().sum())
plt.hist(df["polarity"],color="r",bins=10)
plt.title("reviews distribution")
plt.xlabel("sentiments")
plt.ylabel("count")
plt.show()
print("the number of negative reviews : ",df[df["polarity"]==3]["polarity"].count())
print("the number of natural reviews : ",df[df["polarity"]==2]["polarity"].count())
print("the number of positive reviews : ",df[df["polarity"]==1]["polarity"].count())
df['text']=df['text'].apply(transform_text)
print(df.head())
df.to_csv("dataset_cleand.csv",index=False)


