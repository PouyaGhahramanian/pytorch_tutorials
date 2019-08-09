
import numpy as np
import time
import logging
import spacy
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from creme import stream

PATH_TO_CSV = 'bbc-text.csv'

class MLPClassifier(nn.Module):
    def __init__(self, features_size = 300, hidden_layers_num = 2, hidden_layers_size = 16, classes_num = 5):
        super(MLPClassifier, self).__init__()
        self.device = torch.device('cpu')
        if(torch.cuda.is_available()):
            logging.info('\tFound Cuda Device. Using GPU...')
            self.device = torch.device('cuda:0')
        self.hidden_layer_1 = nn.Linear(300, hidden_layers_size).to(self.device)
        self.hidden_layer_2 = nn.Linear(hidden_layers_size, hidden_layers_size).to(self.device)
        self.output_layer = nn.Linear(hidden_layers_size, classes_num).to(self.device)

    def forward(self, x):
        x = torch.from_numpy(x).float().to(self.device)
        x = F.relu(self.hidden_layer_1(x))
        x = F.relu(self.hidden_layer_2(x))
        x = F.softmax(self.output_layer(x), dim=0)
        return x

    def predict(self, x):
        #print(x)
        pred = torch.argmax(x).cpu().numpy()
        return pred

if(__name__ == '__main__'):

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    logging.info('\tLoading word embeddings and data streamer...')
    start_time = time.time()
    nlp = spacy.load('en_core_web_md')
    encodings = {'tech':0, 'business':1, 'sport':2, 'entertainment':3, 'politics':4}
    encodings_ = [np.array([0, 0, 0, 0, 1]), np.array([0, 0, 0, 1, 0]), np.array([0, 0, 1, 0, 0]), np.array([0, 1, 0, 0, 0]), np.array([1, 0, 0, 0, 0])]
    types = {"category":str}
    dataset = stream.iter_csv(PATH_TO_CSV, target_name="category", types=types)
    stop_time = time.time()
    elapsed_time = stop_time - start_time
    logging.info('\tFinished in {0} seconds.'.format(elapsed_time))

    mlp = MLPClassifier()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(mlp.parameters(), lr=0.01, momentum=0.9)
    predictions = []
    acc = 0.0
    crr = 0.0
    count = 0.0
    data_x = []
    data_y = []
    classes = [0, 1, 2, 3, 4]

    train_start = time.time()

    logging.info('\tGetting data instance embeddings using creme streamer and GloVe model, then training MLP Classifier.')
    for (i, (X, y)) in enumerate(dataset):

        data_instance = []
        X_embedding = np.array([nlp(X['text']).vector])
        y_encodding = np.array([encodings[y]]).ravel()
        #y_encodding_ = torch.from_numpy(y_encodding).to(mlp.device)
        y_encodding_ = torch.from_numpy(np.asarray(encodings_[y_encodding[0]])).long().to(mlp.device)
        data_instance.append(X_embedding)
        data_instance.append(y_encodding)
        data_x.append(np.asarray(X_embedding))
        data_y.append(np.asarray(y_encodding))

        optimizer.zero_grad()
        output = mlp.forward(X_embedding)
        prediction = mlp.predict(output)
        predictions.append(prediction)
        print(output.view(1, -1))
        print(y_encodding_.view(1, -1))
        print(y_encodding)
        loss = criterion(output.view(1, -1), y_encodding_.view(1, -1))
        loss.backward()
        optimizer.step()
        count += 1
        print(prediction, y_encodding)
        if prediction == y_encodding[0]:
            crr +=1
        acc = crr / count
        logging.info('\tAccuracy after training on {0} data instances: {1}'.format(i, acc))

    train_end = time.time()
    train_time = train_end - train_start
    logging.info('\tTraining time: {0}'.format(train_time))
