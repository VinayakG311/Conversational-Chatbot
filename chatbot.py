import random
import json
import pickle
import numpy as np
import nltk

from nltk.stem import WordNetLemmatizer
from keras.models import load_model

lemmatizer = WordNetLemmatizer()
intents = json.loads(open('Data.json').read())
words=pickle.load(open('words.pkl','rb'))
classes=pickle.load(open('classes.pkl','rb'))

model=load_model('chatbotmodel.model')

def clean_up_sentence(sentence):
    sentence_words =nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words
def bag_of_words(sentence):
    sentence_words=clean_up_sentence(sentence)
    bag=[0]*len(words)
    for w in sentence_words:
        for i,word in enumerate(words):
            if word==w:
                bag[i]=1
    return np.array(bag)
def predict_class(sentence):
    bag_words=bag_of_words(sentence)
    res=model.predict(np.array([bag_words]))[0]

    err_thresh=0.25
    result=[[i,r] for i,r in enumerate(res) if r>err_thresh]

    result.sort(key=lambda x : x[1],reverse=True)

    results=[]
    for r in result:
        results.append({'intent':classes[r[0]],'probability':str(r[1])})
    return results

def get_response(intents_list,intents_json):
    tags=intents_list[0]['intent']
    listofintent = intents_json['intents']

    for i in listofintent:
        if i['tag']==tags:
            result=random.choice(i['responses'])
            break
    return result



while True:
    message=input()
    ints=predict_class(message)

    res=get_response(ints,intents)
    print(res)
