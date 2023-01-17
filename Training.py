import random
import json
import pickle
import numpy as np
import nltk

from nltk.stem import WordNetLemmatizer
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD

#
lemmatizer = WordNetLemmatizer()
intents = json.loads(open('Data.json').read())

words = []
classes = []
documents = []
ignore_letters = ['?', '.', ',', '!']
for intent in intents['intents']:
    for pattern in intent["patterns"]:
        word_l = nltk.word_tokenize(pattern)
        words.extend(word_l)
        documents.append((word_l, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])


words = [lemmatizer.lemmatize(word=word) for word in words if word not in ignore_letters]
words = sorted(set(words))
classes=sorted(set(classes))
pickle.dump(words,open("words.pkl","wb"))
pickle.dump(classes,open("classes.pkl","wb"))
training =[]
output=[]
output_empty = [0]*len(classes)
for document in documents:
    bag=[]
    word_patterns = document[0]
    word_patterns=[lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    for word in words:
        if word in word_patterns:
            bag.append(1)
        else:
            bag.append(0)
    output_row=list(output_empty)
    output_row[classes.index(document[1])]=1

    training.append(bag)
    output.append(output_row)

training=np.array(training)
output=np.array(output)
model = Sequential()
model.add(Dense(128,input_shape=(len(training[0]),),activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(output[0]),activation='softmax'))
sgd=SGD(lr=0.01,decay=1e-6,momentum=0.9,nesterov=True)
model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])
hist=model.fit(training,output,epochs=200,batch_size=5,verbose=1)
model.save("chatbotmodel.model",hist)
#
# train_x = list(training[:,0])
# train_y = list(training[:,1])
# print(train_y)