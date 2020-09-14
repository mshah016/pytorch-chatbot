import random
import json
import torch
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

# check if GPU (cuda) is available, if its not, then simply use CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 



with open('greeting_train.json', 'r') as f:
    greetings = json.load(f)

FILE = 'data.pth'
data = torch.load(FILE)

input_size = data['input_size']
hidden_size = data['hidden_size']
output_size = data['output_size']

all_words = data['all_words']
tags = data['tag']
model_state = data['model_state']

model = NeuralNet(input_size, hidden_size, output_size).to(device) #push model to device
model.load_state_dict(model_state)
model.eval()

bot_name = 'Fred'
print("Let's chat! Type 'quit' to exit")

while True:
    sentence = input('You: ')
    if sentence == "quit":
        break
    
    sentence = tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X)

    output = model(X)
    _, predicted = torch.max(output, dim = 1)
    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim = 1)
    prob = probs[0][predicted.item()]

    if prob.item() > 0.75:
        for greet in greetings['greetings']:
            if tag == greet["tag"]:
                print(f"{bot_name}: {random.choice(greet['responses'])}")
    else:
        print(f"{bot_name}: I'm sorry, I do not understand")