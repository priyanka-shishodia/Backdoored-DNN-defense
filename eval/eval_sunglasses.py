import repair 
import sys
import h5py
import cv2
import keras
from random import choice
import numpy as np
import datetime

def data_loader(filepath):
    data = h5py.File(filepath, 'r')
    x_data = np.array(data['data'])
    y_data = np.array(data['label'])
    x_data = x_data.transpose((0,2,3,1))

    return x_data, y_data

def data_preprocess(x_data):
    return x_data/255

def main():
    data_filename = str(sys.argv[1])

    x_test, y_test = data_loader(data_filename)
    #x_test = data_preprocess(x_test)

    model_GoodNet = keras.models.load_model('../models/repaired_nets/model_GoodNet_sun.h5')
    model_BadNet = keras.models.load_model('../models/sunglasses_bd_net.h5')

    y_hat = []
    sequence = [i for i in range(len(repair.valid_x))]

    for i in range(len(x_test)):
        if i%500 == 0:
            print("***Completion: " + str(round((i/len(x_test))*100, 2)) + "%, Time: " + str(datetime.datetime.now()))
        newimages_test = []
        validimages_test = []
        for j in range(repair.numoverlays):
            validimages_test.append(repair.valid_x[choice(sequence)])
            newimages_test.append(data_preprocess(cv2.addWeighted(x_test[i],1,validimages_test[j],1,0,dtype=cv2.CV_64FC3)))

        newimagesnda_test = np.asarray(newimages_test)
        uniquevals = len(np.unique(np.argmax(model_BadNet.predict(newimagesnda_test), axis=1)))
        # print(uniquevals)
        # print(repair.threshold)
        if uniquevals <= repair.threshold: #if poisoned
            y_hat.append(1283)
        else:
            x_test_nest = np.array([x_test[i]])
            x_test_nest = data_preprocess(x_test_nest)
            clean_label_p = np.argmax(model_GoodNet.predict(x_test_nest), axis=1)
            y_hat.append(clean_label_p[0])

    poisonrate = 0
    cleanacc = 0
    incorrect = 0
    totclean = 0.00000001
    for i in range(len(y_hat)):
        if y_hat[i] == 1283:
            poisonrate = poisonrate + 1
        else:
            totclean = totclean + 1
            if y_hat[i] == y_test[i]:
                cleanacc = cleanacc + 1
            else:
                incorrect = incorrect + 1
    
    print(str(poisonrate) + " Poisoned images detected. This is " + str((poisonrate/len(y_hat))*100) + "% of the total images")
    print(str((cleanacc/totclean)*100) + "% of detected-clean images have been predicted correctly. This is " + str((cleanacc/len(y_hat))*100) + "% of the total images")
    print(str((incorrect/totclean)*100) + "% of detected-clean values have been predicted incorrectly. This is " + str((incorrect/len(y_hat))*100) + "% of the total images")

if __name__ == '__main__':
    main()
