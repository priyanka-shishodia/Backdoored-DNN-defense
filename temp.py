import sys
import os
import h5py
import numpy as np
import imageio


def data_loader(filepath):
    data = h5py.File(filepath, 'r')
    x_data = np.array(data['data'])
    y_data = np.array(data['label'])
    x_data = x_data.transpose((0,2,3,1))

    return x_data, y_data

def main():
    print(os.getcwd())

if __name__ == '__main__':
    main()


#ctrl + / to comment group
#poisoned images
sequence = [i for i in range(len(valid_x))]
model_BadNetSTRIP = keras.models.load_model('sunglasses_bd_net.h5')
print("Poisoned Image Check In Progress")
poison_yhat = []
for i in range(len(poison_x)):

    print(poison_x[i])
    poison_xi = data_preprocess(poison_x[i])
    print(poison_xi)
    imageio.imwrite("poisonres.png",poison_xi.astype(np.uint8))
    x_test = cv2.imread("poisonres.png")
    hf = h5py.File('cache.h5', 'w')
    hf.create_dataset('data', data=x_test)
    hf.close()
    x_test = data_loader2('cache.h5')
    x_test = data_preprocess(x_test)
    
    if i%500 == 0:
        print("Poison: " + str(round((i/len(poison_x))*100, 2)) + "%, Time: " + str(datetime.datetime.now()))
    newimages_poison = []
    newimages_poison2 = []
    validimages_poison = []
    for j in range(numoverlays):
        validimages_poison.append(valid_x[choice(sequence)])
        newimages_poison.append(data_preprocess(cv2.addWeighted(poison_x[i],1,validimages_poison[j],1,0,dtype=cv2.CV_64FC3)))
        newimages_poison2.append(data_preprocess(cv2.addWeighted(x_test,1,validimages_poison[j],1,0,dtype=cv2.CV_64FC3)))

    newimagesnda_poison = np.asarray(newimages_poison)
    uniquevals = len(np.unique(np.argmax(model_BadNetSTRIP.predict(newimagesnda_poison), axis=1)))
    
    newimagesnda_poison2 = np.asarray(newimages_poison2)
    uniquevals2 = len(np.unique(np.argmax(model_BadNetSTRIP.predict(newimagesnda_poison2), axis=1)))

    print(str(uniquevals) + " " + str(uniquevals2))

print("")
trueneg = round((sum(poison_yhat)/len(poison_yhat))*100, 2)
print("True Neg: " + str(trueneg) + "%")
