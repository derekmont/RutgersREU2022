#!/usr/bin/env python
# coding: utf-8

# In[ ]:
#######################################################################################
#   Author: Derek Montanez                                                            #
#   Work: Implementation of 'Pruning Filters for Efficient Convnets'                  #
#   Affiliation: DIMACS Rutgers University REU                                        #
#   Command Line Arguments: dataset, learning_rate, epoch1, epoch2, prune_percentage  #
#                                                                                     #
#######################################################################################



import tensorflow as tf
from tensorflow import keras
import numpy as np
import sys

# In[ ]:
###############################################
#                                             #
# Testing for faulty command line arguments   #
#                                             #
###############################################

if (sys.argv[1] != "fashion") and (sys.argv[1] != "cifar10"):
    sys.exit("Wrong Dataset: Please use MNIST Fashion (fashion) or CIFAR10 (cifar10)")
elif float(sys.argv[2]) <= 0: 
    sys.exit("Learning Rate is Not Applicable: Use positive learning rate")
elif int(sys.argv[3]) < 1:
    sys.exit("Batch Size is Not Applicable: Use Larger Batch Size")
elif int(sys.argv[4]) < 1 or  int(sys.argv[5]) < 1:
    sys.exit("An Epoch Size is Not Applicable: Please use larger epoch")
elif float(sys.argv[6]) < 0.0 or float(sys.argv[6]) >= 1.0:
    sys.exit("Percentage of Filters is Not Applicable")


print("Dataset is ", sys.argv[1])
print("Learning Rate is ", sys.argv[2])
print("Batch Size is ", sys.argv[3])
print("First Epoch is ", sys.argv[4])
print("Second Epoch is ", sys.argv[5])
print("Percentage of Filters to be Pruned ", sys.argv[6])


##############################################################
#                                                            #
# Setting Correct values fro x and y test values, as well as #
# image size and channel number, this is dependent on        #
# dataset used.                                              #
#                                                            #
##############################################################

if sys.argv[1] == 'fashion':
    (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
    x_train = x_train / 255
    x_test = x_test/ 255
    input_channels = 1
    img_size = 28
    
    if int(sys.argv[5]) < int(sys.argv[4]): 
        print("Value of epoch too small")
elif sys.argv[1]  == 'cifar10':
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    x_train = x_train / 255
    x_test = x_test/ 255
    input_channels = 3
    img_size = 32
    
    if int(sys.argv[5]) < int(sys.argv[4]): 
        print("Value of epoch too small")
else:
    print("Dataset not applicable")
    sys.exit()

# In[ ]:

###############################################################
#                                                             #
# Function for when the user does not want to prune the model #
# and wants to train the model                                #
#                                                             #
###############################################################

def modelForZeroPercentage(imageSz, channelNumber, xTrain, xTest, yTrain, yTest):

    print("Testing for baseline model")


    input_sh = tf.keras.Input(shape=(img_size, img_size, input_channels))
    x = tf.keras.layers.Conv2D(32, (3, 3))(input_sh)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)

    x = tf.keras.layers.Conv2D(64, (3, 3))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(1280, activation = 'relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    output = tf.keras.layers.Dense(10, activation = 'softmax')(x)

    model4 = tf.keras.Model(inputs = input_sh, outputs = output)

    model4.summary()

    model4.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = float(sys.argv[2])), loss=tf.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])

    model4.fit(xTrain, yTrain, batch_size=int(sys.argv[3]), epochs=int(sys.argv[4]))

    model4.fit(xTrain, yTrain, batch_size=int(sys.argv[3]), epochs=int(sys.argv[5]))

    model4.evaluate(xTest, yTest, batch_size=int(sys.argv[3]))




if float(sys.argv[6]) == 0.0:
    modelForZeroPercentage(img_size, input_channels,  x_train, x_test, y_train, y_test)


if float(sys.argv[6]) == 0.0:
    sys.exit("Cannot Proceed")



percentage = float(sys.argv[6])

FilterArray = [] 
WeightArray = []
BiasArray = []
ListOfIndexes = []

##############################################################
#                                                            #
# The 2 Functions Build A Model, of the SCNNB neural netwok. #
# Will be used when our pruned model is initialized.         #
#                                                            #
##############################################################
def tail2(input_tail):
    x = tf.keras.layers.Flatten()(input_tail)
    x = tf.keras.layers.Dense(1280, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    return tf.keras.layers.Dense(10, activation='softmax')(x)
def model_constructor2(F, W, img_size, input_channels):
    input1 = tf.keras.Input(shape=(img_size, img_size, input_channels))
    
    x = tf.keras.layers.Conv2D(F[0], (3, 3))(input1)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    
    x = tf.keras.layers.Conv2D(F[1], (3, 3), activation='linear')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    
    output = tail2(x)
    
    return tf.keras.Model(inputs=input1, outputs=output)


# In[ ]:


input_sh = tf.keras.Input(shape=(img_size, img_size, input_channels))
x = tf.keras.layers.Conv2D(32, (3, 3))(input_sh)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.ReLU()(x)
x = tf.keras.layers.MaxPooling2D((2, 2))(x)

x = tf.keras.layers.Conv2D(64, (3, 3))(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.ReLU()(x)
x = tf.keras.layers.MaxPooling2D((2, 2))(x)

x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(1280, activation = 'relu')(x)
x = tf.keras.layers.Dropout(0.5)(x)
output = tf.keras.layers.Dense(10, activation = 'softmax')(x)

model3 = tf.keras.Model(inputs = input_sh, outputs = output)


model3.summary()


# In[ ]:


if float(sys.argv[2]) < 0 and int(sys.arg[3]) < 0 and int(sys.arg[4]) < 0:
    print("One of your hyperparameters is not applicable")
else:
    
    print("Now Training Original Model")
    
    model3.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = float(sys.argv[2])), loss=tf.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])

    model3.fit(x_train, y_train, batch_size=int(sys.argv[3]), epochs=int(sys.argv[4]))


# In[ ]:


import math

#######################################################################################################             
#                                                                                                     #
# Input: Model that we want to prune (model), and % of filters to prune from each layer (percentage)  #
#                                                                                                     #
# Output: Array of the number of filters in each layer of convolutional layer after pruning           #
#         Array of the shapes of the weight matrices for the filters                                  #
#         Array of the new biases that will be used to initialize the model                           #
#                                                                                                     #
#######################################################################################################

def pruneNetwork(model, percentage):
    for i in range(len(model.layers)):
        if isinstance(model.layers[i], tf.keras.layers.Conv2D):
            indexesToPrune = calculateSumOfWholeLayer(model.layers[i], math.ceil(model.layers[i].filters * percentage))
            ListOfIndexes.append(indexesToPrune)
            FilterArray.append(model.layers[i].filters - len(indexesToPrune))
            newWeightMatrix = setNetworkData(model.layers[i].get_weights()[0], indexesToPrune)
            newBiasArray = setNetworkDataBias(model.layers[i].get_weights()[1], indexesToPrune)
            WeightArray.append(newWeightMatrix)
            BiasArray.append(newBiasArray)
    
    return FilterArray, WeightArray, BiasArray


# In[ ]:


#####################################################################################################################
#                                                                                                                   #
# Input: Convolutional layer (convolutional), Array of the indexes of filters that will be pruned (indexesToPrune)  #
#                                                                                                                   #
# Output: Returns a reshaped convolutional layer                                                                    #
#                                                                                                                   #
#####################################################################################################################

def changeChannelsInNextLayer(convolutional, indexesToPrune):
    arrToDelete = np.array(indexesToPrune)
    
    returnConv = np.delete(convolutional, arrToDelete, axis=2)
    
    return returnConv


# In[ ]:

########################################################
#                                                      #
# Input: A convolutional layers filter (filter)        #
#                                                      #
# Output: The sum of the magnitude of the filter (num  #
#                                                      #
########################################################

def calculateSumOfFilter(filter):
    num = 0
    for row in range(len(filter)):
        for col in range(len(filter[0])):
            num += sum(abs(filter[row][col]))
    return num


# In[ ]:


###################################################################################
#                                                                                 #
# Input: Convolutional Layer (conv), Number of filters to delete (numberToPrune)  #
#                                                                                 #
# Output: A list of the indicies of filters to prune (listOfSmallestIndicies      #
#                                                                                 #
###################################################################################

def calculateSumOfWholeLayer(conv, numberToPrune):
    list1 = []
    for i in range(0, conv.filters):
        sumOfFilter = calculateSumOfFilter(conv.get_weights()[0][:, :, :, i])
        list1.append(sumOfFilter)
    

    listOfSmallestIndicies = np.argsort(list1)[:numberToPrune]
    
    
    return listOfSmallestIndicies


# In[ ]:


# In[ ]:


#########################################################################################
#                                                                                       #
# Input: A convolutional Layers Biases (biases), Indexes of biases to remove (indexes)  #
#                                                                                       #
# Output: a new bias configuration for the pruned network (returnConv                   #
#                                                                                       #
#########################################################################################

def setNetworkDataBias(biases, indexes):
    arrToDelete = np.array(indexes)
    
    returnConv = np.delete(biases, arrToDelete, axis=0)
        
    return returnConv


# In[ ]:
#############################################################################################################################
#                                                                                                                           #
# Input: Weight Matrix of current convolutional layer (tensot), List of filters to remove in convolutional layer (indexes)  #
#                                                                                                                           #
# Output: New Weught Matrix for convolutional layer (returnConv)                                                            #
#                                                                                                                           #
#############################################################################################################################

def setNetworkData(tensor, indexes):
    arrToDelete = np.array(indexes)
    
    print("Before Pruning")
    print(tensor.shape)

    # make a new set of weights for the convolution
    returnConv = np.delete(tensor, arrToDelete, axis=3)

    print("After Pruning")
    print(returnConv.shape)
        
    return returnConv


# In[ ]:


l1, l2, l3 = pruneNetwork(model3, percentage)

newModel = model_constructor2(l1, l2, img_size, input_channels)

print("This is the summary of the pruned model")

newModel.summary()


# In[ ]:
##############################################################################################################################################################
#                                                                                                                                                            #
# Input: WeightMatrix accumulated so far (weightMatrix, List for all layers, conatining indexes of filters to prune (listofIndicies, original model (model)  #
#                                                                                                                                                            #
# Output: A final weight matrix to be initialized in the final pruned network (finalWeightMatrix)                                                            #
#                                                                                                                                                            #
##############################################################################################################################################################

def configureWeightMatrix(weightMatrix, listofIndicies, model):
    finalWeightMatrix = []
    for i in range(0, len(weightMatrix)):
        if i == 0:
            finalWeightMatrix.append(model.layers[1].get_weights()[0])
        else:
            finalWeightMatrix.append(changeChannelsInNextLayer(weightMatrix[i], listofIndicies[i - 1]))
    return finalWeightMatrix


# In[ ]:


finalWeight = configureWeightMatrix(WeightArray, ListOfIndexes, newModel)


# In[ ]:


finalModel = model_constructor2(l1, finalWeight, img_size, input_channels)


# In[ ]:

k = 0

finalModel.layers[1].set_weights([finalWeight[0], BiasArray[0]])
finalModel.layers[5].set_weights([finalWeight[1], BiasArray[1]])


# In[1]:

print("Will now begin retraining and testing pruned model")

finalModel.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = float(sys.argv[2])), loss=tf.losses.SparseCategoricalCrossentropy(),metrics=['accuracy'])

finalModel.fit(x=x_train, y=y_train, batch_size=int(sys.argv[3]), epochs=int(sys.argv[5]))

finalModel.evaluate(x=x_test, y=y_test, batch_size=int(sys.argv[3]))


# In[ ]:




