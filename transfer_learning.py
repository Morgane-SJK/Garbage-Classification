"""
===============================================================================================================
Test of various pre-trained classification models on the Garbage dataset
In order to use this script, you first have to split the dataset downloaded from Kaggle using `split_dataset.py`
===============================================================================================================
"""

#IMPORTS
import time
import itertools
import os
import numpy as np
import pandas as pd
import sys
import getopt

#Plot of graphs
import matplotlib.pyplot as plt

#Creating and managing image classification models with Keras
from tensorflow import keras
#Access to the images to classify, which are distributed in different folders
from keras.preprocessing.image import ImageDataGenerator
#Different types of layers for CNN
from keras.layers import Flatten, Dense, Activation
#Creation of the model architecture with Model
from keras.models import Model 
#Early stopping
from keras.callbacks import EarlyStopping
#Use of registered models
from keras.models import load_model

#Import of 11 pre-trained models
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input as preprocess_vgg16

from keras.applications.vgg19 import VGG19
from keras.applications.vgg19 import preprocess_input as preprocess_vgg19

from keras.applications.resnet import ResNet50
from keras.applications.resnet import preprocess_input as preprocess_resnet50

from keras.applications.xception import Xception
from keras.applications.xception import preprocess_input as preprocess_xception

from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input as preprocess_inception_v3

from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.inception_resnet_v2 import preprocess_input as preprocess_inception_resnet_v2

from keras.applications.mobilenet import MobileNet
from keras.applications.mobilenet import preprocess_input as preprocess_mobilenet

from keras.applications.densenet import DenseNet121, DenseNet201
from keras.applications.densenet import preprocess_input as preprocess_densenet

from keras.applications.nasnet import NASNetLarge, NASNetMobile
from keras.applications.nasnet import preprocess_input as preprocess_nasnet

#Model performance metrics with Sklearn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


#GLOBAL VARIABLES

#Mandatory argument
modelName = ""

#dataset information
inputFolder = "./Garbage_train_test_val/"
outputFolder = "./Results/"
labels = []
nClasses = 6
img_width = 0
img_height = 0
input_shape = (0, 0, 3)

#Hyper-parameters
batch_size = 32
epochs_nb = 65

dataAugmentation = True
earlyStoppingPatience = 0

saveModel = True
saveConfusionMatrix = True

modelInputSize =   {"VGG16":             [224,224], #[width,height]
                    "VGG19":             [224,224],
                    "ResNet50":          [224,224],
                    "Xception":          [299,299],
                    "InceptionV3":       [299,299],
                    "InceptionResNetV2": [299,299],
                    "MobileNet":         [224,224],
                    "DenseNet121":       [224,224],
                    "DenseNet201":       [224,224],
                    "NASNetLarge":       [331,331],
                    "NASNetMobile":      [224,224]}

train_generator      = None
test_generator       = None
validation_generator = None

model    = None
history  = None
executionTime = 0

def stringToBoolean(string):
    return string.lower() in ("true", "t", "yes", "y", "oui", "o", "1")

def getArguments(): 
    
    global modelName
    global modelInputSize
    global img_width
    global img_height
    global input_shape
    global inputFolder
    global outputFolder
    global labels
    global nClasses
    global batch_size
    global epochs_nb
    global dataAugmentation
    global earlyStoppingPatience
    global saveModel
    global saveConfusionMatrix

    argsName = ["help", "modelName", "inputFolder", "outputFolder", "batch_size", "epochs_nb", "dataAugmentation", "earlyStoppingPatience", "saveModel", "saveConfusionMatrix"]

    helpMessage = ''.join(["\nYou must use the script as follow :\n",
                           "python transfer_learning.py --modelName=\"\"\n\n",
                           "ALL OPTIONS:\n",
                           "--modelName: name of the pre-trained model you want to test. It must be a model of this list : ['VGG16', 'VGG19', 'ResNet50', 'Xception', 'InceptionV3', 'InceptionResNetV2', 'MobileNet', 'DenseNet121', 'DenseNet201', 'NASNetLarge', 'NASNetMobile'].\n",
                           "[OPTIONAL]\n",
                           "--inputFolder: path of the Garbage_train_test_val folder. It has been created with the split_dataset.py script.\n",
                           "--outputFolder: path of the folder which will contain the saved models, confusion matrix...\n",
                           "--batch_size: integer which represent the batch size used by the model. By default: 32\n",
                           "--epochs_nb: integer which represents the number of epochs you want to use to train your model. By default: 65\n",
                           "--dataAugmentation: boolean which indicates if you want to use Data Augmentation to improve the performance of the models. By default: True.\n",
                           "--earlyStoppingPatience: integer which represent the number of epochs you want to wait before stopping the model training when this one stop improving. By default: 0 --> it means that you don't want to use early stopping, and let the model training during all the epochs.\n",
                           "--saveModel: boolean which indicates if you want to save the trained model or not. It will be saved in h5 format, in the outputFolder. By default: True\n",
                           "--saveConfusionMatrix: boolean which indicates if you want to save the confusion matrix. By default: False"])

    try:
        opts, args = getopt.getopt(sys.argv[1:],"",[arg+"=" if arg!="help" else arg for arg in argsName])
    
    except getopt.GetoptError as a:
        print(a)
        print("\nError Command Line Argument : python transfer_learning.py --modelName=\"\"\nFor Further information : python transfer_learning.py --help\n")
        sys.exit()

    options = [opt for opt, arg in opts]

    #Check if the modelName option is present
    if ("--modelName" not in options) and ("--help" not in options):
        print("\nError Command Line Argument : python transfer_learning.py --modelName=\"\"\nFor Further information : python transfer_learning.py --help\n")
        sys.exit()

    for opt, arg in opts:

        if   opt == "--help":
            print(helpMessage)
            sys.exit()

        elif opt == "--modelName":
            if arg in ['VGG16', 'VGG19', 'ResNet50', 'Xception', 'InceptionV3', 'InceptionResNetV2', 'MobileNet', 'DenseNet121', 'DenseNet201', 'NASNetLarge', 'NASNetMobile']:
                modelName = arg
                img_width  = modelInputSize[modelName][0]
                img_height = modelInputSize[modelName][1]
                input_shape = (img_width, img_height, 3)
            else :
                print("\nThe model you specified is not available in this script. You must choose a model part of this list :  ['VGG16', 'VGG19', 'ResNet50', 'Xception', 'InceptionV3', 'InceptionResNetV2', 'MobileNet', 'DenseNet121', 'DenseNet201', 'NASNetLarge', 'NASNetMobile']\n")
                sys.exit()

        elif opt == "--inputFolder":
            inputFolder=arg if arg.endswith('/') else ''.join([arg, '/'])
            
        elif opt == "--outputFolder":
            outputFolder=arg if arg.endswith('/') else ''.join([arg, '/'])

        elif opt == "--batch_size":
            batch_size = int(arg)

        elif opt == "--epochs_nb":
            epochs_nb = int(arg)

        elif opt == "--dataAugmentation":
            dataAugmentation = stringToBoolean(arg)
            
        elif opt == "--earlyStoppingPatience":
            earlyStoppingPatience = int(arg)

        elif opt == "--saveModel":
            saveModel=stringToBoolean(arg)

        elif opt == "--saveConfusionMatrix":
            saveConfusionMatrix=stringToBoolean(arg)
    
    labels = os.listdir(f"{inputFolder}train")
    nClasses = len(labels) #6
    outputFolderExist = os.path.exists(outputFolder)
    if not outputFolderExist:
        # Create a new directory because it does not exist 
        os.makedirs(outputFolder)


def modelSettings():

    global modelName
    global img_width
    global img_height

    if (modelName == "VGG16"):
        preprocessing = preprocess_vgg16

    elif (modelName == "VGG19"):
        preprocessing = preprocess_vgg19

    elif (modelName == "ResNet50"):
        preprocessing = preprocess_resnet50

    elif (modelName == "Xception"):
        preprocessing = preprocess_xception

    elif (modelName == "InceptionV3"):
        preprocessing = preprocess_inception_v3

    elif (modelName== "InceptionResNetV2"):
        preprocessing = preprocess_inception_resnet_v2

    elif (modelName == "MobileNet"):
        preprocessing = preprocess_mobilenet

    elif (modelName== "DenseNet121" or modelName== "DenseNet201"):
        preprocessing = preprocess_densenet
    
    elif (modelName== "NASNetLarge" or modelName== "NASNetMobile"):
        preprocessing = preprocess_nasnet

    return preprocessing


def makeGenerators():

    global modelName
    global img_width
    global img_height
    global inputFolder
    global dataAugmentation
    global batch_size
    global train_generator
    global test_generator
    global validation_generator

    preprocessing = modelSettings()

    if (dataAugmentation):
        train_datagen = ImageDataGenerator( preprocessing_function = preprocessing,
                                            width_shift_range=0.2, 
                                            height_shift_range=0.2, 
                                            horizontal_flip=True, 
                                            vertical_flip=True,
                                            rotation_range = 90,
                                            brightness_range = [0.8, 1.2],
                                            zoom_range = [0.8, 1],
                                            fill_mode='nearest')
    else:
        train_datagen = ImageDataGenerator(preprocessing_function = preprocessing)        

    test_datagen = ImageDataGenerator(preprocessing_function = preprocessing)

    train_generator = train_datagen.flow_from_directory(directory=r'./{}train'.format(inputFolder),
                                                        target_size=(img_width, img_height),
                                                        color_mode='rgb',
                                                        batch_size=batch_size,
                                                        class_mode='categorical',
                                                        shuffle=True,
                                                        seed=42)
    test_generator = test_datagen.flow_from_directory(directory=r'./{}test'.format(inputFolder),
                                                        target_size=(img_width, img_height),
                                                        color_mode='rgb',
                                                        batch_size=batch_size,
                                                        class_mode='categorical',
                                                        shuffle=False)
    validation_generator = test_datagen.flow_from_directory(directory=r'./{}val'.format(inputFolder),
                                                            target_size=(img_width, img_height),
                                                            color_mode='rgb',
                                                            batch_size=batch_size,
                                                            class_mode='categorical',
                                                            shuffle=False)


def architecture():
  
    global modelName
    global model
    global input_shape
    global nClasses

    if modelName == "VGG16":
        model = VGG16(include_top = False, weights = 'imagenet', input_shape = input_shape)

    elif modelName == "VGG19":
        model = VGG19(include_top = False, weights = 'imagenet', input_shape = input_shape)
    
    elif modelName == "ResNet50":
        model = ResNet50(include_top = False, weights = 'imagenet', input_shape = input_shape)
 
    elif modelName == "Xception":
        model = Xception(include_top = False, weights = 'imagenet', input_shape = input_shape)

    elif modelName == "InceptionV3":
        model = InceptionV3(include_top = False, weights = 'imagenet', input_shape = input_shape)    

    elif modelName == "InceptionResNetV2":
        model = InceptionResNetV2(include_top = False, weights = 'imagenet', input_shape = input_shape)

    elif modelName == "MobileNet":
        model = MobileNet(include_top = False, weights = 'imagenet', input_shape = input_shape)  

    elif modelName == "DenseNet121":
        model = DenseNet121(include_top = False, weights = 'imagenet', input_shape = input_shape)

    elif modelName == "DenseNet201":
        model = DenseNet201(include_top = False, weights = 'imagenet', input_shape = input_shape)    

    elif modelName == "NASNetLarge":
        model = NASNetLarge(include_top = False, weights = 'imagenet', input_shape = input_shape)

    elif modelName == "NASNetMobile":
        model = NASNetMobile(include_top = False, weights = 'imagenet', input_shape = input_shape)  
    
    else :
        print("\nThe model you specified is not available in this script. You must choose a model part of this list :  ['VGG16', 'VGG19', 'ResNet50', 'Xception', 'InceptionV3', 'InceptionResNetV2', 'MobileNet', 'DenseNet121', 'DenseNet201', 'NASNetLarge', 'NASNetMobile']\n")
        sys.exit()

    #Convolutive part : we keep the parameters of the convolutive layers
    for layer in model.layers:
        layer.trainable = False
        lastLayer = model.output

    x = Flatten()(lastLayer)

    #Classification part : we add a fully connected layer and a last one with the good number of classes
    #Only this part will be trained by the model
    x = Dense(256)(x)
    x = Activation('relu')(x)
    x = Dense(nClasses)(x)
    x = Activation('softmax')(x)

    model = Model(model.input, x)


def compile_train():

    global model
    global history
    global executionTime
    global earlyStoppingPatience
    global saveModel
    global modelName

    model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.RMSprop(learning_rate=2e-5), metrics=['accuracy'])

    if earlyStoppingPatience > 0:
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=earlyStoppingPatience)
        cb_list=[es]
    else:
        cb_list=[]

    startTime=time.time()
    history = model.fit(train_generator, epochs = epochs_nb, validation_data  = validation_generator, callbacks=cb_list)
    executionTime = (time.time() - startTime)

    if (saveModel):
        model.save(f'./{outputFolder}{modelName}.h5')


def plot_accuracy_loss():
    """
    Plot of 2 graphs : one which present the accuracy on train and validation set, and the second one with the loss
    """

    global history
    global outputFolder
    global modelName

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,5))

    #Accuracy
    accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']
    ax1.plot(range(len(accuracy)), accuracy, label='train')
    ax1.plot(range(len(val_accuracy)), val_accuracy, label='validation')
    ax1.legend()
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Training and Validation Accuracy')

    #Loss
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    ax2.plot(range(len(loss)), loss, label='train')
    ax2.plot(range(len(val_loss)), val_loss, label='validation')
    ax2.legend()
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Loss')
    ax2.set_title('Training and Validation Loss')

    fig.savefig(''.join([outputFolder, modelName, "_Accuracy_Loss.png"]))


def plot_confusion_matrix(cm,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    Display the confusion matrix on the test set
    """

    global labels
    global outputFolder
    global modelName

    fig = plt.figure(figsize=(10,10))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)
    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    fig.savefig(''.join([outputFolder, modelName, "_Confusion_Matrix.png"]))
    plt.close()


def study_model():
    """
    Return information on the model performances
    """

    global model
    global test_generator
    global executionTime
    global saveConfusionMatrix

    #Training duration
    print(f"Training duration : {round(executionTime, 2)} secondes")

    #Performance on the test set
    preds = model.predict(test_generator)  # preds are proba for each class
    preds_classes = [i.argmax() for i in preds]
    y_true = test_generator.classes
    acc_test = accuracy_score(y_true, preds_classes)
    prec_test = precision_score(y_true, preds_classes, average='weighted')
    rec_test = recall_score(y_true, preds_classes, average='weighted')
    f1_test = f1_score(y_true, preds_classes, average='weighted')
    results = pd.DataFrame([[acc_test, prec_test, rec_test, f1_test]], columns=['accuracy', 'precision', 'recall', 'f1-score'])
    results.to_csv(f'./{outputFolder}{modelName}.csv')

    #Confusion matrix
    if (saveConfusionMatrix):
      cm = confusion_matrix(y_true, preds_classes)
      plot_confusion_matrix(cm)


if __name__ == "__main__":
    
    getArguments()
    makeGenerators()
    architecture()
    compile_train()
    plot_accuracy_loss()
    study_model()

    print(f"\nDONE")