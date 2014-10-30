__author__ = 'Schmidtz'

# Back-Propagation Neural Networks for Power consumption prediction
# Dev by Jong-Min Yu @ GIST Ph.D Candidate
#
#
# Written in Python.  See http://www.python.org/
# Placed in the public domain.
# Neil Schemenauer <nas@arctrix.com>

import math
import random
import numpy as np
import matplotlib.pyplot as plt
import string

random.seed(0)

# calculate a random number where:  a <= rand < b
def rand(a, b):
    return (b-a)*random.random() + a

# Make a matrix (we could use NumPy to speed this up)
def makeMatrix(I, J, fill=0.0):
    m = []
    for i in range(I):
        m.append([fill]*J)
    return m

# our sigmoid function, tanh is a little nicer than the standard 1/(1+e^-x)
def sigmoid(x):
    value = math.tanh(x)
    #print value
    return value

# derivative of our sigmoid function, in terms of the output (i.e. y)
def dsigmoid(y):
    return 1.0 - y**2

class NN:
    def __init__(self, ni, nh, no):
        # number of input, hidden, and output nodes
        self.ni = ni + 1 # +1 for bias node
        self.nh = nh
        self.no = no

        # activations for nodes
        self.ai = [1.0]*self.ni
        self.ah = [1.0]*self.nh
        self.ao = [1.0]*self.no

        # create weights
        self.wi = makeMatrix(self.ni, self.nh)
        self.wo = makeMatrix(self.nh, self.no)
        # set them to random vaules
        for i in range(self.ni):
            for j in range(self.nh):
                self.wi[i][j] = rand(-0.2, 0.2)
        for j in range(self.nh):
            for k in range(self.no):
                self.wo[j][k] = rand(-2.0, 2.0)

        # last change in weights for momentum
        self.ci = makeMatrix(self.ni, self.nh)
        self.co = makeMatrix(self.nh, self.no)

    def update(self, inputs):
        if len(inputs) != self.ni-1:
            raise ValueError('wrong number of inputs')

        # input activations
        for i in range(self.ni-1):
            #self.ai[i] = sigmoid(inputs[i])
            self.ai[i] = inputs[i]

        # hidden activations
        for j in range(self.nh):
            sum = 0.0
            for i in range(self.ni):
                sum = sum + self.ai[i] * self.wi[i][j]
            self.ah[j] = sigmoid(sum)

        # output activations
        for k in range(self.no):
            sum = 0.0
            for j in range(self.nh):
                sum = sum + self.ah[j] * self.wo[j][k]
            self.ao[k] = sigmoid(sum)

        return self.ao[:]


    def backPropagate(self, targets, N, M):
        if len(targets) != self.no:
            raise ValueError('wrong number of target values')

        # calculate error terms for output
        output_deltas = [0.0] * self.no
        for k in range(self.no):
            error = targets[k]-self.ao[k]
            output_deltas[k] = dsigmoid(self.ao[k]) * error

        # calculate error terms for hidden
        hidden_deltas = [0.0] * self.nh
        for j in range(self.nh):
            error = 0.0
            for k in range(self.no):
                error = error + output_deltas[k]*self.wo[j][k]
            hidden_deltas[j] = dsigmoid(self.ah[j]) * error

        # update output weights
        for j in range(self.nh):
            for k in range(self.no):
                change = output_deltas[k]*self.ah[j]
                self.wo[j][k] = self.wo[j][k] + N*change + M*self.co[j][k]
                self.co[j][k] = change
                #print N*change, M*self.co[j][k]

        # update input weights
        for i in range(self.ni):
            for j in range(self.nh):
                change = hidden_deltas[j]*self.ai[i]
                self.wi[i][j] = self.wi[i][j] + N*change + M*self.ci[i][j]
                self.ci[i][j] = change

        # calculate error
        error = 0.0
        for k in range(len(targets)):
            error = error + 0.5*(targets[k]-self.ao[k])**2
        return error


    def test(self, patterns):
        cfile = open('./consoution.dat','w')
        for p in patterns:
            temp =  self.update(p)
            if max(temp)==temp[0]:
                print 0
                cfile.write(str(0))
            elif max(temp)==temp[1]:
                print 1
                cfile.write(str(1))
            elif max(temp)==temp[2]:
                print 2
                cfile.write(str(2))
            elif max(temp)==temp[3]:
                print 3
                cfile.write(str(3))
            elif max(temp)==temp[4]:
                print 4
                cfile.write(str(4))
            elif max(temp)==temp[5]:
                print 5
                cfile.write(str(5))
            elif max(temp)==temp[6]:
                print 6
                cfile.write(str(6))
            else:
                print 7
                cfile.write(str(7))
            cfile.write('\n')
        cfile.close()

    def weights(self):
        print('Input weights:')
        for i in range(self.ni):
            print(self.wi[i])
        print()
        print('Output weights:')
        for j in range(self.nh):
            print(self.wo[j])

    def train(self, patterns,label_dat, iterations=200, N=0.1, M=0.01):
        efile = open('error_rate.dat','w')
        wfile = open('weight.dat','w')
        # N: learning rate
        # M: momentum factor
        dat_error = np.zeros((iterations,1),dtype=float)
        error_count  =0
        countt = 0;
        for i in range(iterations):
            temp_error = 0.0
            for p in patterns:
                inputs = p
                targets = label_dat[countt,:]
                self.update(inputs)
                temp_error = temp_error + self.backPropagate(targets, N, M)
                countt = countt+1
            countt=0
            print('error %-.5f' % temp_error)
            dat_error[error_count] = temp_error
            error_count =  error_count +1
            efile.write(str(temp_error))


        wfile.write('input-to-hidden weight\n')
        for i in range(self.ni):
            for j in range(self.nh):
                wfile.write(str(self.wi[i][j]))

        wfile.write('hidden-to-output weight\n')
        for j in range(self.nh):
            for k in range(self.no):
                 wfile.write(str(self.wo[j][k]))

        plt.figure()
        plt.plot(dat_error,'r.-')
        plt.show()
        efile.close()
        wfile.close()



def Prediction_test(dat,label_dat,test_dat):
    # Teach network XOR function
    pat = dat
    test_pat = test_dat

    # create a network with two input, two hidden, and one output nodes
    n = NN(7, 10, 7)
    # train it with some patterns
    n.train(pat,label_dat)
    # test it
    print ('processing is finished')
    n.test(test_pat)

def Regularization(filename, p_dat_file,weather_dat,label_dat,c_count):
    w2_file = open(filename,'w')
    min_value = 869970
    max_value = 1597120
    r1 = (max_value-min_value)/7
    b_count = 0;
    c_count = c_count-1
    p_line = p_dat_file.readline()
    while p_line:
        p_split_dat = p_line.split('\t')
        for i in range(7):
            weather_dat[b_count,i] = float(p_split_dat[i+1])
            w2_file.write(str(p_split_dat[i+1]))

        # the maximum power consumption value is 869970
        # the minimum power consumption value is 1597120
        temp_temp = p_split_dat[8].split('\n')
        temp = int(temp_temp[0])
        #print temp
        if temp <= min_value+r1:
            label_dat[b_count] = [1,0,0,0,0,0,0]
        if temp > min_value+r1 and temp <= min_value+r1*2:
            label_dat[b_count] = [0,1,0,0,0,0,0]
        if temp > min_value+2*r1 and temp <= min_value+r1*3:
            label_dat[b_count] = [0,0,1,0,0,0,0]
        if temp > min_value+3*r1 and temp <= min_value+r1*4:
            label_dat[b_count] = [0,0,0,1,0,0,0]
        if temp > min_value+4*r1 and temp <= min_value+r1*5:
            label_dat[b_count] = [0,0,0,0,1,0,0]
        if temp > min_value+5*r1 and temp <= min_value+r1*6:
            label_dat[b_count] = [0,0,0,0,0,1,0]
        if temp > min_value+6*r1 and temp <= min_value+r1*8:
            label_dat[b_count] = [0,0,0,0,0,0,1]
        print label_dat[b_count]
        b_count = b_count +1
        if b_count==c_count:
            break
        p_line = p_dat_file.readline()
    p_dat_file.close()
    #regularization for back propagation
    #divided by each maximum value
    weather_dat[:,0] = weather_dat[:,0]/26.7
    weather_dat[:,1] = weather_dat[:,1]/30.7
    weather_dat[:,2] = weather_dat[:,2]/23.8
    weather_dat[:,3] = weather_dat[:,3]/18.2
    weather_dat[:,4] = weather_dat[:,4]/5.2
    weather_dat[:,5] = weather_dat[:,5]/3
    weather_dat[:,6] = weather_dat[:,6]/100
    return weather_dat,label_dat


if __name__ == '__main__':
    dat_file = open('./data.dat')
    write_file = open('./re_data.dat','w')
    dat_line = dat_file.readline()
    c_count = 0;
    while dat_line:
        #data set processing
        dat_line = dat_file.readline()
        split_dat =  dat_line.split('\t')
        c_count = c_count +1
        for i in range(len(split_dat)):
            write_file.write('\t')
            write_file.write(split_dat[i])
    write_file.close()
    dat_file.close()

    weather_dat = np.zeros((c_count,7),dtype=float)
    label_dat = np.zeros((c_count,7),dtype=float)
    test_w_dat = np.zeros((273,7),dtype = float)
    t_label_dat = np.zeros((273,7),dtype = float)

    p_dat_file = open('./re_data.dat')
    weather_dat,label_dat = Regularization('./re_regularize.dat',p_dat_file,weather_dat,label_dat,c_count)
    t_dat_file = open('./temp_file.dat')
    t_line = t_dat_file.readline()
    t_count = 0
    while t_line:
        ts_dat = t_line.split('\t')
        for i in range(len(ts_dat)-1):
             test_w_dat[t_count,i] = float(ts_dat[i])
        t_count = t_count+1
        t_line = t_dat_file.readline()


    Prediction_test(weather_dat,label_dat,test_w_dat)
    p_dat_file.close()
    t_dat_file.close()
