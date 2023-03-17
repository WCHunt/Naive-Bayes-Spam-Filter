# Naive-Bayes-Spam-Filter

## Description
This is a Naive Bayes classifier for spam detection in Go. It uses a simple bag-of-words model to train on a dataset of spam and non-spam (ham) emails and then classify new emails as either spam or ham. The algorithm assumes that each word in an email is independent of the other words, which is clearly not true, hence the "naive" in the name of the algorithm.

This project was originally a school assignment that the I enjoyed, written in C++, but was later ported to Go to practice writing Go code and to experiment with parallelization.

The project solves the problem of classifying a given text file as either spam or ham. The model is trained on a dataset of known spam and real text files and then applied to new a new text file to determine it's classification.
 

The classifier works as follows:

1. Read in a dataset of text files labeled as real and spam.

2. Tokenize the text files by splitting them into individual words and storing the frequency of each word in the dataset.

3. Calculate the prior probabilities of each class (spam and real) and the conditional probabilities of each word given each class.

4. To classify a new text file, tokenize it, calculate the log-probability of it belonging to each class using the prior and conditional probabilities, and choose the class with the highest probability.

To use the classifier, clone the repository and follow the instructions in the README.md file.
