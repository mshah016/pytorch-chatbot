import nltk
import numpy as np
# nltk.download('punkt')
from nltk.stem.porter import PorterStemmer
# create stemmer - stemmed string, or base of the word
stemmer = PorterStemmer()

def tokenize(sentence):
    """ takes in sentence and returns tokenized sentence, an array of strings """
    return nltk.word_tokenize(sentence)

def stem(word):
    """ takes in word and returns the base of that word """
    # make word lowercase
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence, all_words):
    """ compares tokenized_sentence to all_words; if the words in the tokenized_sentence is available in all_words, the words are converted to 1s and 0s in the bag """
    tokenized_sentence = [stem(w) for w in tokenized_sentence]

    # create bag - initialize with zeros (np.zeros) with length of all_words
    bag = np.zeros(len(all_words), dtype=np.float32)
    for idx, w in enumerate(all_words): #gives both index and word
        if w in tokenized_sentence:
            bag[idx] = 1.0 #bag element with this particular index = 1.0
    return bag



# # test tokenization 
# a = "who are you?"
# print(a)
# a = tokenize(a)
# print(a)

# # test stemming function
# b = ["Organize", "organizes", "organizing", "organized"]
# print(b)
# stemmed_words = [stem(w) for w in b]
# print(stemmed_words)

# # test bag_of_words_function
# sentence = ["hello", "how", "are", "you"]
# words = ["hi", "hello", "I", "you", "bye", "thank", "cool"]
# bag = bag_of_words(sentence, words)
# print(bag)