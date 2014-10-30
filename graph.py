__author__ = 'Schmidtz'

import matplotlib.pyplot as plt
import matplotlib.legend as legend
import numpy as np

def Regularization(filename, p_dat_file,weather_dat,label_dat,c_count):
    w2_file = open(filename,'w')
    min_value = 869970
    max_value = 1597120
    r1 = (max_value-min_value)/7
    b_count = 0;
    c_count = c_count-1
    p_line = p_dat_file.readline()
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
            label_dat[b_count] = 0
        if temp > min_value+r1 and temp <= min_value+r1*2:
            label_dat[b_count] = 1
        if temp > min_value+2*r1 and temp <= min_value+r1*3:
            label_dat[b_count] = 2
        if temp > min_value+3*r1 and temp <= min_value+r1*4:
            label_dat[b_count] = 3
        if temp > min_value+4*r1 and temp <= min_value+r1*5:
            label_dat[b_count] = 4
        if temp > min_value+5*r1 and temp <= min_value+r1*6:
            label_dat[b_count] = 5
        if temp > min_value+6*r1 and temp <= min_value+r1*8:
            label_dat[b_count] = 6
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
    error_file =  open('./error_rate.dat')
    test_dat_file = open('./test_dat.dat')
    reproc_file = open('./re_test.dat','w')
    test_line = test_dat_file.readline()
    c_count = 0;
    while test_line:
        test_split_dat = test_line.split('\t')
        c_count = c_count +1
        for i in range(len(test_split_dat)):
            reproc_file.write('\t')
            reproc_file.write(test_split_dat[i])
        test_line = test_dat_file.readline()

    test_dat_file.close()
    reproc_file.close()

    w_dat = np.zeros((c_count,7),dtype=float)
    wl_dat = np.zeros((c_count,1),dtype=int)
    p_dat_file = open('./re_test.dat')
    w_dat, wl_dat= Regularization('./test_Reg.dat', p_dat_file,w_dat,wl_dat,c_count)
    test_dat_file.close()
    reproc_file.close()

    temp_file = open('temp_file.dat','w')
    for i in range(c_count):
        for j in range(7):
            temp_file.write(str(w_dat[i,j]))
            temp_file.write('\t')
        temp_file.write(str(wl_dat[i]))
        temp_file.write('\n')
    temp_file.close()

    plt.figure()
    gg,=plt.plot(wl_dat,'b.-')

    con_file = open('./consoution.dat')
    da = con_file.readline()
    co = 0
    drta = np.zeros((273,1),dtype = float)
    while da:
        drta[co] = float(da)
        da = con_file.readline()
        co = co+1
    tt,=plt.plot(drta,'r.-')
    plt.legend([gg,tt],['Read data','Prediction Data'],loc=3)
    plt.ylabel('Power load category')
    plt.xlabel('Days')
    plt.show()






