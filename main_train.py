import matplotlib
import matplotlib.pyplot as plt
from Intents.Intents import new_intents
from ChatBot_v2.generate import DataGenerator
from ChatBot_v2.trainer import ModelTrainer

DataGenerator.generate(new_intents, dir_path='./data')

mt = ModelTrainer('./data/train_x_df.csv', './data/train_y_df.csv')
history = mt.startTraining('./model')

def plot_graphs(history, string):
    plt.plot(history.history[string])
    # plt.plot(history.history['val_'+string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    # plt.legend([string, 'val_'+string])
    plt.savefig(r'C:\\Users\\Smit\\Desktop\\Final Year Project\\{0}.png'.format(string))

plot_graphs(history, 'accuracy')
plot_graphs(history, 'loss')



