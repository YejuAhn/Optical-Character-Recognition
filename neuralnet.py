#Goal : implements an optical character recognizer using a one hidden layer neural net with sigmoid activation 
#learn the parameters of the model on training data
#report the cross-entropy at the end of each epoch on both train and test data,
#and at the end of training write out its predictions and error rates on both datasets. (mean train error and test error)

import sys
import csv
import numpy as np
import math
import matplotlib.pyplot as plt


def store_data(input_file):
    f  = open(input_file, 'r')
    input_f = csv.reader(f)
    X = []
    Y = []
    for ith_row in input_f: #N
        y_i = np.zeros(10, dtype=int)
        y_i[int(ith_row[0])] = 1 
        x_i = []
        x_i.append(1) #bias term added
        for i in range(1, len(ith_row)): #M
            x_i.append(int(ith_row[i])) 
        y_i = y_i.T
        y_i.shape = (y_i.size, 1)
        x_i = np.array(x_i)
        x_i = x_i.T
        x_i.shape = (len(x_i), 1)
        X.append(x_i)
        Y.append(y_i)
    Y = np.array(Y)
    X = np.array(X) #array of matrices of X and Y
    return [Y, X]


def linear_forward(x, y):
    return np.dot(y, x)


def sigmoid_forward(a):
    return 1 / (1 + np.exp(-a))


def softmax_forward(b):
    return np.exp(b)/sum(np.exp(b))


def cross_entropy_forward(y, y_hat): #loss function: -yTlog(y_hat)
    return -(np.dot(y.T, np.log(y_hat)))


def nn_forward(train_x, train_y, alpha, beta):
    a = linear_forward(train_x, alpha)
    z = sigmoid_forward(a)
    #add bias term to z 
    z = np.append([1], z)
    z.shape = (z.size ,1)
    b = linear_forward(z, beta)
    y_hat = softmax_forward(b)
    J = cross_entropy_forward(train_y, y_hat)
    return [train_x, a, b, z, y_hat, J]


#combined cross-entropy and softmax layer together
def cross_entropy_softmax_backward(y, y_hat, J, b, g_J):
    return np.multiply(g_J, (y_hat - y))


def linear_backward(a, alpha, b, g_b):
    g_alpha = np.dot(g_b, np.transpose(a))
    g_a = np.dot(np.transpose(alpha), g_b)
    return (g_alpha, g_a)

def sigmoid_backward(a, z, g_z):
    return np.multiply(np.multiply(g_z, z), (1 - z))


def nn_backward(x, y, alpha, beta, o):
    x, a, b, z, y_hat, J = o[0], o[1], o[2], o[3], o[4], o[5]
    g_J = 1 #base case: gradient always one
    g_b = cross_entropy_softmax_backward(y, y_hat, J, b, g_J)
    beta_star = beta[:,1:]
    g_beta, g_z = linear_backward(z, beta_star, b, g_b)
    z_star = np.delete(z, 0)
    z_star.shape = (z_star.size, 1)
    g_a = sigmoid_backward(a, z_star, g_z)
    g_alpha, g_x = linear_backward(x, alpha, a, g_a)
    return (g_alpha, g_beta)


def predict(data, alpha, beta, output):
    Y = data[0]
    X = data[1]
    error = 0.0
    out_file = open(output, 'w')
    for i in range(len(Y)):
        o = nn_forward(X[i], Y[i], alpha, beta)
        y_hat = np.argmax(o[-2])
        txt = str(y_hat) + "\n"
        out_file.write(txt)
        if y_hat != np.argmax(Y[i]):
            error += 1.0
    return error / len(Y)



def plot_analysis(epochs, mean_train_cross_entropy, mean_test_cross_entropy, learn_rate):
    # plot analysis
    fig = plt.figure()
    fig.suptitle('learn_rate: 0.001', fontsize=16)
    #train
    plt.errorbar(epochs, mean_train_cross_entropy, label='Train Data')
    #validation
    plt.errorbar(epochs, mean_test_cross_entropy, label='Test Data')
    plt.legend(loc='lower right')
    plt.show()


def sgd(train_data, test_data, num_epoch, alpha, beta, learn_rate, metrics_out, train_out, test_out):
    train_Y = train_data[0]
    train_X = train_data[1]
    test_Y = test_data[0]
    test_X = test_data[1]
    metrics_out_file = open(metrics_out, 'w')
    epochs = []
    mean_train_cross_entropys = []
    mean_test_cross_entropys = []
    for i in range(num_epoch):
        epochs.append(i)
        for j in range(len(train_Y)):
            #compute nn layers
            o = nn_forward(train_X[j], train_Y[j], alpha, beta)
            #compute gradients via backprop
            g_alpha, g_beta = nn_backward(train_X[j], train_Y[j], alpha, beta, o)
            #update 
            alpha = alpha - np.multiply(learn_rate, g_alpha)
            beta = beta - np.multiply(learn_rate, g_beta)

        #train mean cross-entropy
        train_cross_entropy = 0.0
        for k in range(len(train_Y)):
            train_o = nn_forward(train_X[k], train_Y[k], alpha, beta)
            train_cross_entropy += train_o[-1]
        mean_train_cross_entropy = (train_cross_entropy / len(train_Y)).item(0)

        #test mean cross-entropy
        test_cross_entropy = 0.0
        for p in range(len(test_Y)):
            test_o = nn_forward(test_X[p], test_Y[p], alpha, beta)
            test_cross_entropy += test_o[-1]
        mean_test_cross_entropy = (test_cross_entropy / len(test_Y)).item(0)

        txt = "epoch=" + str(i + 1) + " crossentropy(train): " + str(mean_train_cross_entropy) + "\n"
        txt += "epoch=" + str(i + 1) + " crossentropy(test): " + str(mean_test_cross_entropy) + "\n"
        metrics_out_file.write(txt)
        mean_train_cross_entropys.append(mean_train_cross_entropy)
        mean_test_cross_entropys.append(mean_test_cross_entropy)

    #prediction
    train_error = predict(train_data, alpha, beta, train_out)
    test_error = predict(test_data, alpha, beta, test_out)
    error_txt = "error(train): " + str(train_error) + "\n" + "error(test): " + str(test_error)
    metrics_out_file.write(error_txt)
    return (alpha, beta, mean_train_cross_entropys, mean_test_cross_entropys, epochs)


if __name__ == '__main__':
    #path to the input .tsv file
    train_input = sys.argv[1]
    test_input = sys.argv[2]
    #output .labels files
    train_out = sys.argv[3]
    test_out = sys.argv[4]
    #metrics output
    metrics_out = sys.argv[5]
    #epoch
    num_epoch = int(sys.argv[6])
    #number of hidden units
    D = int(sys.argv[7])

    # D_1 = int(sys.argv[7])
    # D_2 = int(sys.argv[8])
    # D_3 = int(sys.argv[9])
    # D_4 = int(sys.argv[10])
    # D_5 = int(sys.argv[11])
    # D = [D_1, D_2, D_3,D_4, D_5]
    # #init_flag
    # init_flag =  int(sys.argv[12])
    # learn_rate = float(sys.argv[13])

    init_flag =  int(sys.argv[8])
    # # learning rate
    learn_rate = float(sys.argv[9])
    # lr_1 = float(sys.argv[9])
    # lr_2 = float(sys.argv[10])
    # lr_3 = float(sys.argv[11])
    # lr = [lr_1, lr_2, lr_3]

    #store data
    train_data = store_data(train_input)
    test_data = store_data(test_input)

    train_X = train_data[1]
    M_with_bias = len(train_X[0]) #number of input features, including bias
    K = 10 #number of label classes manually set [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]     

    #initialization
    if init_flag == 1:
        #random uniform distribution [-0.1, 0.1]
        alpha = np.random.uniform(-0.1, 0.2, size = (D, M_with_bias)) #(D x M + 1)
        beta = np.random.uniform(-0.1, 0.2, size = (K, D + 1)) #(K x D + 1)
        #initialize bias terms to be 0.0 
        alpha[:,:1] = 0.0
        beta[:,:1] = 0.0
        o = sgd(train_data, test_data, num_epoch, alpha, beta, learn_rate, metrics_out, train_out, test_out)
        mean_train_cross_entropys = o[-3]
        mean_test_cross_entropys = o[-2]
        epochs = o[-1]
    elif init_flag == 2:
        #zeros
        alpha = np.zeros((D, M_with_bias)) #(D x M + 1)
        beta = np.zeros((K, D + 1)) #(K x D + 1)
        sgd(train_data, test_data, num_epoch, alpha, beta, learn_rate, metrics_out, train_out, test_out)
    else:
        print("Not valid")

    plot_analysis(epochs, mean_train_cross_entropys, mean_test_cross_entropys, learn_rate)




    


