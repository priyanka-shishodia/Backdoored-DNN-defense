#IMPORTS
import keras
import keras.backend as K
from keras import initializers
from keras.models import load_model
from keras.utils import plot_model
from keras import models
import tensorflow as tf
import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt
from random import seed
from random import choice
import cv2
import datetime
from shutil import copyfile, move
from scipy.stats import rankdata
import math
import imageio
print("Imports Complete")




#-----------------------------------------------------------------
#FUNCTIONS

#Function takes in data and formats it properly for evaluation, outputting the data and output. Taken from eval.py
def data_loader(filepath):
    data = h5py.File(filepath, 'r')
    x_data = np.array(data['data'])
    y_data = np.array(data['label'])
    x_data = x_data.transpose((0,2,3,1))

    return x_data, y_data

#normalizes data. Taken from eval.py
def data_preprocess(x_data):
    return x_data/255

def evalcustommodel(clean_data_filename, bd_model):
    clean_data_filename = str(clean_data_filename)
    x_test, y_test = data_loader(clean_data_filename)
    x_test = data_preprocess(x_test)

    #bd_model = keras.models.load_model(model_name)
    clean_label_p = np.argmax(bd_model.predict(x_test), axis=1)
    class_accu = np.mean(np.equal(clean_label_p, y_test))*100
    #print('Classification accuracy:', class_accu)
    return class_accu

#Calculations for bottom X percent of weights
def calc_bottom_X_percent_weight(weights, fraction):
    max = weights[0][0][0][0]
    min = weights[0][0][0][0]
    for i in range(len(weights)):
        for j in range(len(weights[i])):
            for k in range(len(weights[i][j])):
                for m in range(len(weights[i][j][k])):
                    if weights[i][j][k][m] < min:
                        min = weights[i][j][k][m]
                    if weights[i][j][k][m] > max:
                        max = weights[i][j][k][m]
    
    truemin = min+(fraction*(max-min))

  #print("Min Activation: ",min)
  #print("Max Activation: ",max)
  #print("Bottom 5% of Range: ",truemin-min)
  #print("Min: ",truemin)

    return truemin

def clear_min_weights(weights, thresh):
    for i in range(len(weights)):
        for j in range(len(weights[i])):
            for k in range(len(weights[i][j])):
                for m in range(len(weights[i][j][k])):
                    if weights[i][j][k][m] < thresh:
                        weights[i][j][k][m] = 0
    return weights

def get_conv_index(model):
    #getting all indices where layer is convolutional layer
    convindex = []
    for i in range(len(model.layers)):
        layername = str(type(model.get_layer(index=i)))
        if "convolutional" in layername:
            convindex.append(i)
    return convindex

def tuning(model, valid_x, valid_y):
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics = ['accuracy'])

    valid_x_pp = data_preprocess(valid_x)
    history = model.fit(valid_x_pp, valid_y, epochs=1)
    return history

def finepruning(model, x, valid_x, valid_y):
    layer_weights = []
    convindex = get_conv_index(model)

    for i in convindex:
        layer_weights.append(model.layers[i].get_weights()[0])

    min_weights_thr = []
    for i in range(len(convindex)):
        min_weights_thr.append(calc_bottom_X_percent_weight(layer_weights[i], x))

    new_weights = []
    for i in range(len(convindex)):
        new_weights.append(clear_min_weights(layer_weights[i], min_weights_thr[i]))

    map_indices = {}
    for i in range(len(convindex)):
        map_indices[i] = convindex[i]
    weights_biases = [0 for x in range(2)]

    for key in map_indices:
        bias_weights = model.layers[map_indices[key]].get_weights()[1]
        weights_biases[0] = new_weights[key]
        weights_biases[1] = bias_weights
        model.layers[map_indices[key]].set_weights(weights_biases)

    tuning(model, valid_x, valid_y)
    
    return model

def datainit():
  #------------------------------------------------------------------------
  #DATA INITIALIZATION

  #os.chdir(os.path.dirname(sys.argv[0]))

  # validation data (all good)
  valid_x, valid_y = data_loader('../data/clean_validation_data.h5')
  #imageio.imwrite("cleanres.png",valid_x[0].astype(np.uint8)) #too lossy
  #print(valid_x.shape, valid_y.shape)
  #plt.imshow(valid_x[0]/255.0) 
  #plt.show()
  #print(valid_y[0])

  # test data (all good)
  test_x, test_y = data_loader('../data/clean_test_data.h5')
  #print(test_x.shape, test_y.shape)
  #plt.imshow(test_x[0]/255.0) 
  #plt.show()
  #print(test_y[0])

  # poisoned data (all bad)
  poison_x, poison_y = data_loader('../data/sunglasses_poisoned_data.h5')
  imageio.imwrite("poisoned_images/poisonres_sunglasses.png",poison_x[0].astype(np.uint8)) #too lossy
  #print(poison_x.shape, poison_y.shape)
  #plt.imshow(poison_x[0]/255.0) 
  #plt.show()
  #print(poison_y[0])

  # anonymous 1 poisoned data (all bad)
  anon1_x, anon1_y = data_loader('../data/anonymous_1_poisoned_data.h5')
  imageio.imwrite("poisoned_images/poisonres_anon1.png",poison_x[0].astype(np.uint8))
  #print(anon1_x.shape, anon1_y.shape)
  #plt.imshow(anon1_x[0]/255.0) 
  #plt.show()
  #print(anon1_y[0])

  # Eyebrows poisoned data (all bad)
  eye_x, eye_y = data_loader('../data/Multi-trigger Multi-target/eyebrows_poisoned_data.h5')
  imageio.imwrite("poisoned_images/poisonres_eyebrows.png",poison_x[0].astype(np.uint8))
  #print(eye_x.shape, eye_y.shape)
  #plt.imshow(eye_x[0]/255.0) 
  #plt.show()
  #print(eye_y[0])

  # Lipstick poisoned data (all bad)
  lip_x, lip_y = data_loader('../data/Multi-trigger Multi-target/lipstick_poisoned_data.h5')
  imageio.imwrite("poisoned_images/poisonres_lipstick.png",poison_x[0].astype(np.uint8))
  #print(lip_x.shape, lip_y.shape)
  #plt.imshow(lip_x[0]/255.0) 
  #plt.show()
  #print(lip_y[0])

  # Sunglasses poisoned data (all bad)
  sun_x, sun_y = data_loader('../data/Multi-trigger Multi-target/sunglasses_poisoned_data.h5')
  imageio.imwrite("poisoned_images/poisonres_mtmtsunglasses.png",poison_x[0].astype(np.uint8))
  #print(sun_x.shape, sun_y.shape)
  #plt.imshow(sun_x[0]/255.0) 
  #plt.show()
  #print(sun_y[0])

  print("Data Initialized")
  return valid_x, valid_y, test_x, test_y, poison_x, poison_y, anon1_x, anon1_y, eye_x, eye_y, lip_x, lip_y, sun_x, sun_y 

  

def repairnet(valid_x, valid_y, test_x, test_y, poison_x, poison_y, anon1_x, anon1_y, eye_x, eye_y, lip_x, lip_y, sun_x, sun_y):
  print("Repairing Networks")

  #------------------------------------------------------------------------
  #SUNGLASSES NET REPAIR

  # loading new instance of model that can be modified
  model_BadNet_sun = load_model('../models/sunglasses_bd_net.h5')
  model_GoodNet_sun = load_model('../models/sunglasses_bd_net.h5')
  #model_BadNet_sun.summary()

  deviation = 20
  prune = 0.05
  poison_target = 15

  acc_test_BadNet_sun = evalcustommodel("../data/clean_test_data.h5", model_BadNet_sun)
  acc_poison_BadNet_sun = evalcustommodel("../data/sunglasses_poisoned_data.h5", model_BadNet_sun)
  acc_cutoff = acc_test_BadNet_sun - deviation
  step_accuracy = acc_cutoff
  print('Accuracy cutoff', acc_cutoff)
  print("Poison cutoff", poison_target)

  while (step_accuracy >= acc_cutoff) and (acc_poison_BadNet_sun >= poison_target):
      model_GoodNet_sun = finepruning(model_GoodNet_sun, prune, valid_x, valid_y)
      step_accuracy = evalcustommodel("../data/clean_test_data.h5", model_GoodNet_sun)
      acc_poison_BadNet_sun = evalcustommodel("../data/sunglasses_poisoned_data.h5", model_GoodNet_sun)
      print('Clean accuracy:', step_accuracy)
      print("Poison accuracy:" + str(acc_poison_BadNet_sun))

  model_GoodNet_sun.save('../models/repaired_nets/model_GoodNet_sun.h5')
  print("Sunglasses Network Repaired (1/4 Complete)")


  #------------------------------------------------------------------------
  #ANONYMOUS1 NET REPAIR

  # loading new instance of model that can be modified
  model_BadNet_anon1 = load_model('../models/anonymous_1_bd_net.h5')
  model_GoodNet_anon1 = load_model('../models/anonymous_1_bd_net.h5')
  #model_BadNet_sun.summary()

  #deviation = 10
  prune = 0.05
  poison_target = 15

  acc_poison_BadNet_anon1 = evalcustommodel("../data/anonymous_1_poisoned_data.h5", model_BadNet_anon1)
  print('Poison cutoff', poison_target)

  while (acc_poison_BadNet_anon1 >= poison_target):
      model_GoodNet_anon1 = finepruning(model_GoodNet_anon1, prune, valid_x, valid_y)
      acc_poison_BadNet_anon1 = evalcustommodel("../data/anonymous_1_poisoned_data.h5", model_GoodNet_anon1)
      print("Poison accuracy:" + str(acc_poison_BadNet_anon1))

  model_GoodNet_anon1.save('../models/repaired_nets/model_GoodNet_anon1.h5')
  print("Anonymous 1 Network Repaired (2/4 Complete)")


  #------------------------------------------------------------------------
  #ANONYMOUS2 NET REPAIR

  # loading new instance of model that can be modified
  model_BadNet_anon2 = load_model('../models/anonymous_2_bd_net.h5')
  model_GoodNet_anon2 = load_model('../models/anonymous_2_bd_net.h5')
  #model_BadNet_sun.summary()

  #deviation = 10
  prune = 0.05
  poison_target = 15

  acc_poison_BadNet_anon2 = evalcustommodel("../data/anonymous_1_poisoned_data.h5", model_BadNet_anon2)
  print('Poison cutoff', poison_target)

  while (acc_poison_BadNet_anon2 >= poison_target):
      model_GoodNet_anon2 = finepruning(model_GoodNet_anon2, prune, valid_x, valid_y)
      acc_poison_BadNet_anon2 = evalcustommodel("../data/anonymous_1_poisoned_data.h5", model_GoodNet_anon2)
      print("Poison accuracy:" + str(acc_poison_BadNet_anon2))

  model_GoodNet_anon2.save('../models/repaired_nets/model_GoodNet_anon2.h5')
  print("Anonymous 2 Network Repaired (3/4 Complete)")


  #------------------------------------------------------------------------
  #MULTI TRIGGER MULTI TARGET NET REPAIR

  # loading new instance of model that can be modified
  model_BadNet_mtmt = load_model('../models/multi_trigger_multi_target_bd_net.h5')
  model_GoodNet_mtmt = load_model('../models/multi_trigger_multi_target_bd_net.h5')
  #model_BadNet_sun.summary()

  #deviation = 10
  prune = 0.05
  poison_target = 15

  acc_poison_BadNet_mtmtsun = evalcustommodel("../data/Multi-trigger Multi-target/sunglasses_poisoned_data.h5", model_BadNet_mtmt)
  acc_poison_BadNet_mtmteye = evalcustommodel("../data/Multi-trigger Multi-target/eyebrows_poisoned_data.h5", model_BadNet_mtmt)
  acc_poison_BadNet_mtmtlip = evalcustommodel("../data/Multi-trigger Multi-target/lipstick_poisoned_data.h5", model_BadNet_mtmt)
  acc_poison_BadNet_mtmt = sum([acc_poison_BadNet_mtmtsun, acc_poison_BadNet_mtmteye, acc_poison_BadNet_mtmtlip])/3.0
  print('Poison cutoff', poison_target)


  while (acc_poison_BadNet_mtmt >= poison_target):
      model_GoodNet_mtmt = finepruning(model_GoodNet_mtmt, prune, valid_x, valid_y)
      acc_poison_BadNet_mtmtsun = evalcustommodel("../data/Multi-trigger Multi-target/mtmtsunglasses_poisoned_data.h5", model_GoodNet_mtmt)
      acc_poison_BadNet_mtmteye = evalcustommodel("../data/Multi-trigger Multi-target/eyebrows_poisoned_data.h5", model_GoodNet_mtmt)
      acc_poison_BadNet_mtmtlip = evalcustommodel("../data/Multi-trigger Multi-target/lipstick_poisoned_data.h5", model_GoodNet_mtmt)
      acc_poison_BadNet_mtmt = sum([acc_poison_BadNet_mtmtsun, acc_poison_BadNet_mtmteye, acc_poison_BadNet_mtmtlip])/3.0
      print("Poison accuracy:" + str(acc_poison_BadNet_mtmt))

  model_GoodNet_mtmt.save('../models/repaired_nets/model_GoodNet_mtmt.h5')
  print("Multi Trigger Multi Target Network Repaired (4/4 Complete)")
  print("Network Repair Complete")

def stripinit(complex_mode, numoverlays, valid_x, test_x, poison_x, anon1_x, eye_x, lip_x, sun_x):
  #------------------------------------------------------------------------
  #STRIP THRESHOLD INITIALIZATION
  print("Beginning STRIP Initialization")

  if(complex_mode != "complex"):
    threshold = 21
    anon1thr = 46
    anon2thr = 30
    MTMTthr = 42
    print("Complete")


    




    return threshold, anon1thr, anon2thr, MTMTthr
  else:
    #initialization
    sequence = [i for i in range(len(valid_x))]
    model_BadNetSTRIP = keras.models.load_model('../models/sunglasses_bd_net.h5')
    model_BadNetAnon1STRIP = keras.models.load_model('../models/anonymous_1_bd_net.h5')
    model_BadNetAnon2STRIP = keras.models.load_model('../models/anonymous_2_bd_net.h5')
    model_BadNetMTMTSTRIP = keras.models.load_model('../models/multi_trigger_multi_target_bd_net.h5')

    #analysis for poisoned and test images

    #notes:
    #overlaying images
    #Alpha and Beta weights are 1 to keep weights the same
    #Gamma is 0 so nothing is added to each value
    #data type is specified since function thinks both input images are of different types

    #test image check
    print("Sunglasses Test Image Check In Progress")
    uniquelabels_test = []
    for i in range(len(test_x)):
      if i%500 == 0:
        print("Test: " + str(round((i/len(test_x))*100, 2)) + "%, Time: " + str(datetime.datetime.now()))
      newimages_test = []
      validimages_test = []
      for j in range(numoverlays):
        validimages_test.append(valid_x[choice(sequence)])
        newimages_test.append(data_preprocess(cv2.addWeighted(test_x[i],1,validimages_test[j],1,0,dtype=cv2.CV_64FC3)))

      newimagesnda_test = np.asarray(newimages_test)
      uniquelabels_test.append(len(np.unique(np.argmax(model_BadNetSTRIP.predict(newimagesnda_test), axis=1))))
    print("Done")

    #poison image check
    uniquelabels_poison = []
    print("Sunglasses Poison Image Test In Progress")
    for i in range(len(poison_x)):
      if i%500 == 0:
        print("Poison: " + str(round((i/len(poison_x))*100, 2)) + "%, Time: " + str(datetime.datetime.now()))
      newimages_poison = []
      validimages_poison = []
      for j in range(numoverlays):
        validimages_poison.append(valid_x[choice(sequence)])
        newimages_poison.append(data_preprocess(cv2.addWeighted(poison_x[i],1,validimages_poison[j],1,0,dtype=cv2.CV_64FC3)))

      newimagesnda_poison = np.asarray(newimages_poison)
      uniquelabels_poison.append(len(np.unique(np.argmax(model_BadNetSTRIP.predict(newimagesnda_poison), axis=1))))
    print("Done")

    #Two standard deviations accounts for most data
    idealtest = np.mean(uniquelabels_test) - 2*(np.std(uniquelabels_test))
    idealpoison = np.mean(uniquelabels_poison) + 2*(np.std(uniquelabels_poison))
    threshold = round((idealtest+idealpoison)/2)

    print("Sunglasses Threshold:", threshold)


    #anonymous 1 poison image check
    uniquelabels_anon1 = []
    print("Anonymous 1 Poison Image Test In Progress")
    for i in range(len(anon1_x)):
      if i%500 == 0:
        print("Anon1: " + str(round((i/len(anon1_x))*100, 2)) + "%, Time: " + str(datetime.datetime.now()))
      newimages_anon1 = []
      validimages_anon1 = []
      for j in range(numoverlays):
        validimages_anon1.append(valid_x[choice(sequence)])
        newimages_anon1.append(data_preprocess(cv2.addWeighted(anon1_x[i],1,validimages_anon1[j],1,0,dtype=cv2.CV_64FC3)))

      newimagesnda_anon1 = np.asarray(newimages_anon1)
      uniquelabels_anon1.append(len(np.unique(np.argmax(model_BadNetAnon1STRIP.predict(newimagesnda_anon1), axis=1))))
    print("Done")

    #Two standard deviations accounts for most data
    anon1thr = round(np.mean(uniquelabels_anon1) + 2*(np.std(uniquelabels_anon1)))

    print("Anonymous 1 Threshold:", anon1thr)



    #anonymous 2 poison image check
    print("Anonymous 2 Poison Image Test In Progress")
    print("No Data Provided For Anonymous 2 Network, Setting Arbitrarily")
    anon2thr = 30

    print("Anonymous 2 Threshold:", anon2thr)


    #MTMT poison image check
    #eyebrows poison image check
    uniquelabels_eye = []
    print("MTMT Eyebrows Poison Image Test In Progress")
    for i in range(len(eye_x)):
      if i%500 == 0:
        print("Eye: " + str(round((i/len(eye_x))*100, 2)) + "%, Time: " + str(datetime.datetime.now()))
      newimages_eye = []
      validimages_eye = []
      for j in range(numoverlays):
        validimages_eye.append(valid_x[choice(sequence)])
        newimages_eye.append(data_preprocess(cv2.addWeighted(eye_x[i],1,validimages_eye[j],1,0,dtype=cv2.CV_64FC3)))

      newimagesnda_eye = np.asarray(newimages_eye)
      uniquelabels_eye.append(len(np.unique(np.argmax(model_BadNetMTMTSTRIP.predict(newimagesnda_eye), axis=1))))
    print("Done")

    #lipstick poison image check
    uniquelabels_lip = []
    print("MTMT Lipstick Poison Image Test In Progress")
    for i in range(len(lip_x)):
      if i%500 == 0:
        print("Lip: " + str(round((i/len(lip_x))*100, 2)) + "%, Time: " + str(datetime.datetime.now()))
      newimages_lip = []
      validimages_lip = []
      for j in range(numoverlays):
        validimages_lip.append(valid_x[choice(sequence)])
        newimages_lip.append(data_preprocess(cv2.addWeighted(lip_x[i],1,validimages_lip[j],1,0,dtype=cv2.CV_64FC3)))

      newimagesnda_lip = np.asarray(newimages_lip)
      uniquelabels_lip.append(len(np.unique(np.argmax(model_BadNetMTMTSTRIP.predict(newimagesnda_lip), axis=1))))
    print("Done")

    #sunglasses poison image check
    uniquelabels_sun = []
    print("MTMT Sunglasses Poison Image Test In Progress")
    for i in range(len(sun_x)):
      if i%500 == 0:
        print("Sun: " + str(round((i/len(sun_x))*100, 2)) + "%, Time: " + str(datetime.datetime.now()))
      newimages_sun = []
      validimages_sun = []
      for j in range(numoverlays):
        validimages_sun.append(valid_x[choice(sequence)])
        newimages_sun.append(data_preprocess(cv2.addWeighted(sun_x[i],1,validimages_sun[j],1,0,dtype=cv2.CV_64FC3)))

      newimagesnda_sun = np.asarray(newimages_sun)
      uniquelabels_sun.append(len(np.unique(np.argmax(model_BadNetMTMTSTRIP.predict(newimagesnda_sun), axis=1))))
    print("Done")

    eyethr = np.mean(uniquelabels_eye) + 2*(np.std(uniquelabels_eye))
    lipthr = np.mean(uniquelabels_lip) + 2*(np.std(uniquelabels_lip))
    sunthr = np.mean(uniquelabels_sun) + 2*(np.std(uniquelabels_sun))

    MTMTthr = round((eyethr + lipthr + sunthr)/3)

    print("MTMT Threshold:", MTMTthr)
    return threshold, anon1thr, anon2thr, MTMTthr



print("Functions Loaded")

#init - don't touch
test_x = None
threshold = None
anon1thr = None
anon2thr = None
MTMTthr = None

#user input
numoverlays = 100

initial_run = ""
if len(sys.argv) >= 2:
  initial_run = str(sys.argv[1])

complex_mode = ""
if len(sys.argv) >= 3:
  complex_mode = str(sys.argv[2])

if initial_run == "init":
  valid_x, valid_y, test_x, test_y, poison_x, poison_y, anon1_x, anon1_y, eye_x, eye_y, lip_x, lip_y, sun_x, sun_y = datainit()
  repairnet(valid_x, valid_y, test_x, test_y, poison_x, poison_y, anon1_x, anon1_y, eye_x, eye_y, lip_x, lip_y, sun_x, sun_y)
  threshold, anon1thr, anon2thr, MTMTthr = stripinit(complex_mode, numoverlays, valid_x, test_x, poison_x, anon1_x, eye_x, lip_x, sun_x)
else:
  valid_x, valid_y, test_x, test_y, poison_x, poison_y, anon1_x, anon1_y, eye_x, eye_y, lip_x, lip_y, sun_x, sun_y = datainit()
  threshold, anon1thr, anon2thr, MTMTthr = stripinit(complex_mode, numoverlays, valid_x, test_x, poison_x, anon1_x, eye_x, lip_x, sun_x)
