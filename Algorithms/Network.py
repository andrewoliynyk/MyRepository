import csv
import random
import pickle
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

plotlist_y = []

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def sigmoid_derivative(x):
    return (1 - sigmoid(x))*sigmoid(x)

class Network(object):

    def __init__(self, sizes):
        self.number_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1)/10 for y in sizes[1:]]
        self.weights = [np.random.randn(x, y)/10
                        for x, y in zip(sizes[1:],sizes[:len(sizes)-1])]

    def save_biases(self):
        with open('biases.txt', 'wb') as fp:
            pickle.dump(self.biases, fp)

    def load_biases(self):
        with open ('biases.txt', 'rb') as fp:
            self.biases = pickle.load(fp)

    def save_weights(self):
        with open('weights.txt', 'wb') as fp:
            pickle.dump(self.weights, fp)

    def load_weights(self):
        with open ('weights.txt', 'rb') as fp:
            self.weights = pickle.load(fp)

    def feedforward(self, a):
        a = a.astype(float)
        for b, w, in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a.reshape((w.shape[1],1))) + b) 
        return a

    def SGD(self, train_data, steps, learn_speed, group_size, test_data):
        global plotlist_y
        plotlist_y.append(self.correct(test_data))
        for i in range(steps):
            random.shuffle(train_data)
            groups = [ train_data[k:k+group_size]
                       for k in range(0, len(train_data), group_size)]
            for group in groups:
                self.update_using_group(group, learn_speed)
            good_answers = self.correct(test_data)
            plotlist_y.append(good_answers/len(test_data)*100) ###
            print("Epoch %s: %s/%s  %s" %
                    (i+1, good_answers, len(test_data),
                    good_answers/len(test_data)*100))

    def update_using_group(self, group, learn_speed):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in group:
            delta_b, delta_w = self.backprop(x, y)
            nabla_b = [nb + db for nb, db in zip(nabla_b, delta_b)]
            nabla_w = [nw + dw for nw, dw in zip(nabla_w, delta_w)]
        self.weights = [w - (learn_speed/len(group))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (learn_speed/len(group))*nb
                        for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        x = x.astype(float)
        y = y.astype(float)
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        z = x
        z_s = [x]
        a_s = []
        for b, w in zip(self.biases, self.weights):
            a = np.dot(w, z.reshape((w.shape[1],1))) + b
            a_s.append(a)
            z = sigmoid(a)
            z_s.append(z)
        delta = (z_s[-1] - y)*sigmoid_derivative(a_s[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, z_s[-2].transpose())
        for i in range(2, self.number_layers):
            a = a_s[-i]
            delta = np.dot(self.weights[-i+1].transpose(), delta)*sigmoid_derivative(a)
            nabla_b[-i] = delta
            nabla_w[-i] = np.dot(delta,
                                 z_s[-i-1].reshape((1,z_s[-i-1].shape[0])))
        return (nabla_b, nabla_w)

    def correct(self, test_data):
        result = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]
        return sum(int(x == int(y)) for (x, y) in result)


def vectorize(i):
    result = np.zeros((10, 1))
    result[i] = 1.0
    return result

def load_train_data(n1, n2):
    x = []
    y = []
    counter = 0
    with open('train.csv', newline='') as csvfile:
        myreader = csv.reader(csvfile)
        for row in myreader:
            if counter >= n1 and counter < n2:
                x.append(list(row)[1:])
                y.append(vectorize(int(list(row)[0])))
            if counter >= n2:
                break
            counter += 1
    x = np.array(x).astype(float)/255
    y = np.array(y).astype(float)
    train_data = np.array(list(zip(x,y)))
    print("Train data: complited")
    return(train_data)

def load_test_data(n1, n2):
    x = []
    y = []
    counter = 0
    with open('train.csv', newline='') as csvfile:
        myreader = csv.reader(csvfile)
        for row in myreader:
            if counter >= n1 and counter < n2:
                x.append(list(row)[1:])
                y.append(list(row)[0])
            if counter >= n2:
                break
            counter += 1
    x = np.array(x).astype(float)/255
    y = np.array(y).astype(float)
    test_data = np.array(list(zip(x,y)))
    print("Test data: complited")
    return(test_data)

def load_validation_data():
    x = []
    with open('test.csv', newline='') as csvfile:
        myreader = csv.reader(csvfile)
        for row in myreader:
            x.append(list(row))
    validation_data = np.array(x).astype(float)/255
    return(validation_data)

train_data = load_train_data(0, 42000)
test_data = load_test_data(32000, 42000)
validation_data = load_validation_data()
net = Network([784,50,10])
net.SGD(train_data,5,1.0,1,test_data)

with open('pred.csv', 'w', newline='') as csvfile:
    spamwriter = csv.writer(csvfile)
    result = [np.argmax(net.feedforward(x)) for x in validation_data]
    spamwriter.writerow(['ImageId','Label'])
    i = 0
    for item in result:
        i += 1
        if i % 100 == 0:
            print(i)
        spamwriter.writerow([i]+[item])
