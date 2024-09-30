import pandas as pd
import pandas as pd
import numpy as np
from string import punctuation
# import re
import nltk
from nltk.corpus import twitter_samples
import random
nltk.download('stopwords')
import string   
from tensorflow.keras.preprocessing.text import Tokenizer                        
from nltk.corpus import stopwords 
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense,Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import seaborn as sns



def tokenizee():
    df = pd.read_csv("Twitter_Data.csv")
    df['clean_text']=df['clean_text'].astype('str')
    df=df.dropna()
    stopwords = stopwords.words('english')
    def transform(text):
       text = text.lower() # converting every characters to lower case
       text = nltk.word_tokenize(text) # tokenizing all the words in the text
       
       # removing the special characters from the text

       y = []
       for i in text:
           if i.isalnum():
               y.append(i)
        
       text = y[:]
       y.clear()
       for i in text:
           if i not in stopwords and i not in string.punctuation:
            y.append(i)
       text = y[:]
       y.clear()
       return " ".join(text) 
    df['clean_text'] = df['clean_text'].apply(transform)
    
    ps = PorterStemmer()
    def stemming (text):
        y=[]
        for i in text.split():
            y.append(ps.stem(i))
        return " ".join(y)
    
    df['clean_text'] = df['clean_text'].apply(lambda x:stemming(x))
        
    df['clean_text'] = df['clean_text'].apply(lambda x:stemming(x))
    df['category'] = [2 if x == -1 else x for x in df['category']]
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(df.clean_text)
    word_index = tokenizer.word_index
    # print(word_index)
    return tokenizer