# from ChatBot.core import Trainer
from Intents.Intents import new_intents
from ChatBot_v2.predictor import ModelPredictor



# mp = ModelPredictor('./data/classes_s.csv', './data/transformedWordDict_s.csv', model_path='./model')
# while True:
#     ip = str(input("USER --> "))
#     pre = mp.getPredictionFor(ip)
#     # printing tags
#     #print(pre)
#     print("JARVIS --> ", mp.getResponseForClass(pre, new_intents))

def loadModel(ip):
    mp = ModelPredictor('./data/classes_s.csv', './data/transformedWordDict_s.csv', model_path='./model')
    # while True:
    # ip = str(input("USER --> "))
    pre = mp.getPredictionFor(ip)
    # printing tags
    #print(pre)
    print("JARVIS --> ", mp.getResponseForClass(pre, new_intents))
    return mp.getResponseForClass(pre, new_intents)
loadModel("Hi There")