
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import time
import spacy
from creme import stream
from torch.autograd import Variable
from sklearn.datasets import make_blobs
from sklearn.preprocessing import OneHotEncoder

class MLP(nn.Module):

    def __init__(self, features_num = 300, hidden_layer_num = 2, hidden_layer_size = 32, classes_num = 5):

        super(MLP, self).__init__()
        self.hidden_layer_num = hidden_layer_num
        self.hidden_layer_size = hidden_layer_size
        self.classes_num = classes_num
        self.features_num = features_num
        self.hidden_layer_1 = nn.Linear(self.features_num, self.hidden_layer_size)
        self.hidden_layer_2 = nn.Linear(self.hidden_layer_size, self.hidden_layer_size)
        self.hidden_layer_3 = nn.Linear(self.hidden_layer_size, self.hidden_layer_size)
        self.output_layer = nn.Linear(self.hidden_layer_size, self.classes_num)

    def forward(self, x):

        output = F.relu(self.hidden_layer_1(x))
        output = F.relu(self.hidden_layer_2(output))
        output = F.relu(self.hidden_layer_3(output))
        output = F.softmax(self.output_layer(output))
        return output

    def predict(self, o):

        return torch.max(o.data, 1)[1].cpu().numpy()

def describe(x):

    #print("Type: {}".format(x.type()))
    print("Shape/size: {}".format(x.shape))
    print("Values: \n{}".format(x))

PATH_TO_CSV = 'bbc-text.csv'

if __name__ == '__main__':

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    '''
    1. Loading dataset as stream using Creme framework.
    '''
    logging.info('\tLoading word embeddings and data streamer...')
    PATH_TO_CSV = 'bbc-text.csv'
    start_time = time.time()
    nlp = spacy.load('en_core_web_md')
    encodings = {'tech':0, 'business':1, 'sport':2, 'entertainment':3, 'politics':4}
    types = {"category":str}
    dataset = stream.iter_csv(PATH_TO_CSV, target_name="category", types=types)
    stop_time = time.time()
    elapsed_time = stop_time - start_time
    logging.info('\tFinished in {0} seconds.'.format(elapsed_time))

    '''
    2. Initializing model.
    '''
    logging.info('\tInitializing model and hyperparameters...')
    start_time = time.time()
    batch_size = 1
    epochs = 1
    learning_rate = 1e-3
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    mlp = MLP(features_num = 300)
    mlp.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(mlp.parameters(), lr = learning_rate)
    stop_time = time.time()
    elapsed_time = stop_time - start_time
    logging.info('\tFinished in {0} seconds.'.format(elapsed_time))

    '''
    3. Training the model.
    '''
    dataset_size = 2225
    logging.info('\tTraining model...')
    train_start_time = time.time()
    for e in range(epochs):
        epoch_start_time = time.time()
        x_batch = []
        y_batch = []
        preds = []
        counter = 0
        counter_2 = 0
        loss = 0
        acc = 0.0
        crr = 0.0
        total = 0.0
        batch_num = 0
        logging_period = 100
        #dataset = stream.iter_csv(PATH_TO_CSV, target_name="category", types=types)
        s_time = time.time()
        for (i, (X, y)) in enumerate(dataset):

            X_embedding = np.array([nlp(X['text']).vector])
            y_encodding = np.array([encodings[y]]).ravel()
            x_batch.append(X_embedding)
            y_batch.append(y_encodding[0])
            counter += 1
            counter_2 += 1
            if counter >= batch_size or i >= dataset_size-1:

                batch_num += 1
                np_input = np.asarray(x_batch).reshape(-1, 300)
                input = torch.from_numpy(np_input).float().to(device)
                input.requires_grad = True
                np_label = np.asarray(y_batch).reshape(counter,)
                label = torch.from_numpy(np_label).long().to(device)
                output = mlp(input).to(device)
                pred = mlp.predict(output)
                preds.append(pred)
                total += counter
                '''
                if pred[0] == y_batch[0] :
                    crr += 1
                '''
                crr += (pred == y_batch).sum()
                '''
                print(output)
                print(label)
                print(pred)
                print(y_batch)
                print((pred == y_batch))
                print((pred == y_batch).sum())
                '''

                optimizer.zero_grad()
                loss = criterion(output, label)
                loss.backward()
                optimizer.step()
                x_batch = []
                y_batch = []
                counter = 0
                if counter_2 >= logging_period :
                    counter_2 = 0
                    batch_end_time = time.time()
                    elapsed_batch_time = int(batch_end_time - s_time)
                    acc = round( (( crr / total ) * 100.0), 2)
                    logging.info('\tEpoch: {0} -- Data instance # : {1} -- Elapsed Time: {2} s \n\t\tLoss: {3} -- Accuracy: {4}'.format(e + 1, i + 1, elapsed_batch_time, loss, acc))

        #pred = predict()
        epoch_stop_time = time.time()
        epoch_elapsed_time = int(epoch_stop_time - epoch_start_time)
        logging.info('\tEpoch: {0} -- Finished in: {1} -- Loss: {2}'.format(e + 1, epoch_elapsed_time, loss))

    train_stop_time = time.time()
    train_elapsed_time = int(train_stop_time - train_start_time)
    logging.info('\tFinished training in {0} seconds.'.format(train_elapsed_time))

    '''
    #input = Variable(torch.randn(1, 300), requires_grad = True).to(device)
    #print(describe(input))
    #np_input = np.random.randn(2, 300)
    #np_input = data_x
    np_input = data_x[:batch_size].reshape(batch_size, -1)
    input = torch.from_numpy(np_input).float().to(device)
    input.requires_grad = True
    #output = Variable(torch.randn(1, 5), requires_grad=True).to(device)
    #print(describe(input))
    #label = Variable(torch.LongTensor(2).random_(5)).to(device)
    #print(describe(label))
    np_label = np.array([data_y[:batch_size]]).reshape(batch_size,)
    label = torch.from_numpy(np_label).long().to(device)
    #print(describe(label))
    output = mlp(input)
    #print(describe(output))

    optimizer.zero_grad()
    loss = criterion(output, label)
    #print(describe(output))
    #print(describe(label))
    print(loss)
    loss.backward()
    optimizer.step()
    output = mlp(input)
    loss = criterion(output, label)
    print(loss)
    '''
