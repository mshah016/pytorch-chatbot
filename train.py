# load json file 
import json
from nltk_utils import tokenize, stem, bag_of_words
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# import model
from model import NeuralNet

# read data
with open('greeting_train.json', 'r') as f:
    greetings = json.load(f)

# print(greetings)

# CREATE TRAINING DATA
all_words = [] #array of all tokenized words
tags = [] #array of all tags in json file 
xy = [] #holds tuple; tokenized words from 'patterns' key and the words from 'tag' key

# loop through greetings
for greet in greetings['greetings']:
    tag = greet['tag'] # tag equals everything in json file with key 'tag'
    tags.append(tag)
    for pattern in greet['patterns']: # pattern equals everything in json file with key 'pattern'
        w = tokenize(pattern) # tokenizes each pattern from the 'pattern' tag
        all_words.extend(w) #use extend because we are appending an array
        xy.append((w, tag)) #append a tuple, tokenized pattern words + its tag

# exclude punctuation
ignore_words = ['?', "!", ".", ","]

# stem all elements of all_words, but exclude the ignore_words
all_words = [stem(w) for w in all_words if w not in ignore_words]
# print(all_words)

# remove duplicates from newly stemmed arrays
all_words = sorted(set(all_words))
tags = sorted(set(tags))
# print(tags)

# create bag of words
X_train = []
y_train = []

# loop over xy array
for (pattern_sentence, tag) in xy:
    # for each tuple in xy, bag runs the bag_of_words function for the pattern sentence and the stemmed version of the all words 
    bag = bag_of_words(pattern_sentence, all_words) #bag: array of floats [0.0, 0.0, 1.0, 0.0]
    X_train.append(bag)

    # variable labels assigned to the index of each tag 
    label = tags.index(tag)
    y_train.append(label) #because we are using pytorch, we need CrossEntropyLoss instead of one-hot here

# convert both training arrays to numpy arrays 
X_train = np.array(X_train)
y_train = np.array(y_train)

# create new dataset
class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(X_train) #states that the number of samples = the length of X_train 
        self.x_data = X_train # set x_data for this class as the X_train dataset
        self.y_data = y_train # set y_data for this class as the y_train dataset

    # to later access dataset with index 
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    # find the length of the dataset
    def __len__(self):
        return self.n_samples

# hyper parameters 
batch_size = 8
hidden_size = 8
output_size = len(tags) #number of different classes or tags that we have
input_size = len(X_train[0]) #length of each bag of words that we created, index 0 because X_train is a list, X_train[0] is the first element 
# print(batch_size)
# print(input_size)
# specify learning rate 
learning_rate = 0.001

# specify number of epochs you want to run
num_epochs = 1000
# print(input_size, len(all_words))
# print(output_size, tags)


dataset = ChatDataset()

# create DataLoader
train_loader = DataLoader(dataset = dataset, batch_size = batch_size, shuffle = True, num_workers = 0)
print(train_loader) 

# create model

# check if GPU (cuda) is available, if its not, then simply use CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
model = NeuralNet(input_size, hidden_size, output_size).to(device) #push model to device

# create loss and optimizer
criterion = torch.nn.CrossEntropyLoss() #measures the amount of loss that is generated
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

# training loop
for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device) #push words to device
        labels = labels.to(device) #push labels to device

        # pass forward
        outputs = model(words)
        # calculate loss with criterion, which gets the predicted loss and the actual labels
        loss = criterion(outputs, labels.long())

        # pass backward and optimizer step

        # first empty gradient
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


    if (epoch + 1) % 100 == 0:
        print(f'epoch {epoch + 1}/{num_epochs}, loss={loss.item():.4f}')

print(f'final loss, loss={loss.item():.4f}')

data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "output_size": output_size,
    "hidden_size": hidden_size,
    "all_words": all_words,
    "tag": tags
}

FILE = "data.pth"
torch.save(data, FILE)

print(f'Training Complete. File saved to {FILE}')