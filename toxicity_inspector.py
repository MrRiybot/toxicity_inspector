import sys, os, re, csv, codecs, numpy as np, pandas as pd
import matplotlib.pyplot as plt
from keras_preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.models import Model
from keras import initializers, regularizers, constraints, optimizers, layers
import pickle
from keras.models import load_model
from spacy_langdetect import LanguageDetector
import spacy
from spacy.language import Language
@Language.factory('language_detector')
def language_detector(nlp, name):
    return LanguageDetector()

class Toxicity_model():
    
    def __init__(self):
        self.arabic_model,self.arabic_tokenizer = self._get_model('arabic_model.h5','arabic_tokenizer.pickle')
        self.english_model,self.english_tokenizer = self._get_model('english_model.h5','english_tokenizer.pickle')
        self.nlp = self._get_langauge_model()
        
    def _get_model(self,model_path,tokenizer_path):
        arabic_model = load_model(model_path)

        # loading
        with open(tokenizer_path, 'rb') as handle:
            arabic_tokenizer = pickle.load(handle)
    
        return arabic_model,arabic_tokenizer

    def _predict_arabic(self,inp,model,tokenizer):
        text_token = tokenizer.texts_to_sequences(inp)
        maxlen = 13 # choosed by the avg
        text_token_pad = pad_sequences(text_token, maxlen=maxlen)

        o = model.predict(text_token_pad)
        return o
    
    def _predict_english(self,inp,model,tokenizer):
        text_token = tokenizer.texts_to_sequences(inp)
        maxlen = 200 # choosed by the avg
        text_token_pad = pad_sequences(text_token, maxlen=maxlen)

        o = model.predict(text_token_pad)
        return o
    
    def _get_langauge_model(self):
        nlp = spacy.load('en_core_web_sm')
        nlp.max_length = 2000000
        nlp.add_pipe('language_detector', last=True)
        return nlp

    def _predict_langauge(self,text,nlp): 
        doc = nlp(text)
        detect_language = doc._.language
        return detect_language

    def _predict_toxicity(self,text,arabic_model,arabic_tokenizer,english_model,english_tokenizer,nlp):
        if self._predict_langauge(text,nlp)['language'] == 'en':
            return 'en',self._predict_english([text],self.english_model,self.english_tokenizer)[0][0]
        elif self._predict_langauge(text,nlp)['language'] in ['ar','fa','ur']:
            return 'ar',self._predict_arabic([text],self.arabic_model,self.arabic_tokenizer)[0][0] 
        else:
            return -1,None
   
    
    def display_prediction(self,text):
        lang,pred = self._predict_toxicity(text,self.arabic_model,self.arabic_tokenizer,self.english_model,self.english_tokenizer,self.nlp)
        if(lang==-1):
            print("model does not take other lang than arabic,english")
            return None
        if(lang=='en'):
            print("text is english")
        elif (lang=='ar'):
            print('text is arabic')
        if(pred>=0.5):
            print("text is TOXIC")
        else:
            print("text is not TOXIC")
    
    def predict(self,text):
        return self._predict_toxicity(text,self.arabic_model,self.arabic_tokenizer,self.english_model,
                                      self.english_tokenizer,self.nlp)
    