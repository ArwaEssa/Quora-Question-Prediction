import streamlit as st
import pickle
import nltk
import numpy as np
from nltk.corpus import stopwords

from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import re
import  string

#model=pickle.load(open('modelnew1.pkl','rb'))
pipe_lr = joblib.load(open('modelnew2.pkl','rb'))
def clean_text (text ):
   stop = nltk.corpus.stopwords.words('english')
    #print(stop)dont 
   stop.append('im')
   stop.append('like')
   stop.append('best')
   stop.append('ever')
   stop.append('im')
   stop.append('dont')
   stop.append('better')
   print(stop)
      # cleaning text
   text = re.sub(r"US", "America", text)
   text = re.sub(r" \w*\d\w*", "", text)
   text = re.sub(r" \d\d\d", "", text)
   text = re.sub(r" \d\d", "", text)
   text = re.sub('\w*\d',' ',text)
   text = re.sub(r" UK ", "England", text)
   text = re.sub(r" J K ", "JK", text)
   text = re.sub(r'“如果不靠欺骗自己，还能靠什么支撑自己走下去”','If you don’t rely on deceiving yourself, what else can you rely on to support yourself',text)
   text = re.sub(r'没出息','unpromising',text)
   text = re.sub(r'シ instead of し','instead of',text)
   text = re.sub(r'’|‘','',text)
   text = re.sub("\'s", "", text)
   text = re.sub(r'\s+[a-zA-Z]\s+', ' ', text)
   text = re.sub(r"([.,!?])", r" \1 ", text)
    # Removing multiple spaces
   text = re.sub(r'\s+', ' ', text)
    # Remove single characters from the start
   text = re.sub(r'\^[a-zA-Z]\s+', ' ', text) 
   text = re.sub(r"(@\[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "", text)
    # Remove all the special characters
   text = re.sub(r'\W', ' ', text)
    #Removing Punctuations
   punctuation=string.punctuation
   text = [word for word in text if word not in punctuation]
   text = ''.join(text)
    #Removing capital letters
   text= text.lower()
    #Removing stopwords
   text = text.split()
   text = [w for w in text if not w in stop]
   text = " ".join(text)
    #lemmatize
   stemmer = WordNetLemmatizer()
   text = [stemmer.lemmatize(word) for word in text]
   text = ''.join(text)
   
   text = text.split()
   text = [w for w in  text if len(w) > 3]
   text =" ".join(text)
    # Return a list of words
   return text

def convert_tonumber(text):
   tfidfconverter = TfidfVectorizer(min_df=5)
   text = tfidfconverter.fit_transform(text)
   return text

def predict_duplecate(text):
    prediction=pipe_lr.predict([text])
    return prediction[0]

def main():
    
    st.title(" Quora Question prediction ")
    Q = st.text_input('Enter your Question:' )
    Text=clean_text(Q)
    #text_array = convert_tonumber(text_array)

    if st.button("Predict"):      
        output=predict_duplecate(Text)
        st.write('The Question Predictis ', output)

if __name__=='__main__':
    main()
