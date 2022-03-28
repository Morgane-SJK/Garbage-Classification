"""
=================================================================================================
Script which allows the comparison of various models which are trained and saved with h5 format
You must have a folder named Models which contains all of these models
=================================================================================================
"""

#IMPORTS
import sys
import pandas as pd

#Managing image classification models with Keras
from tensorflow import keras

#Accès aux images à classifier, qui sont réparties dans différents dossiers
from keras.preprocessing.image import ImageDataGenerator

#Use of registered models
from keras.models import load_model

#Import of 11 preprocessed methods
from keras.applications.vgg16 import preprocess_input as preprocess_vgg16
from keras.applications.vgg19 import preprocess_input as preprocess_vgg19
from keras.applications.resnet import preprocess_input as preprocess_resnet50
from keras.applications.xception import preprocess_input as preprocess_xception
from keras.applications.inception_v3 import preprocess_input as preprocess_inception_v3
from keras.applications.inception_resnet_v2 import preprocess_input as preprocess_inception_resnet_v2
from keras.applications.mobilenet import preprocess_input as preprocess_mobilenet
from keras.applications.densenet import preprocess_input as preprocess_densenet
from keras.applications.nasnet import preprocess_input as preprocess_nasnet

#Model performance metrics with Sklearn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

#GLOBAL VARIABLES
batch_size = 32
inputFolder = "./Garbage_train_test_val/"

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


def modelSettings(modelName):

    global modelInputSize

    img_width = modelInputSize[modelName][0]
    img_height = modelInputSize[modelName][1]

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

    else :
        print("The model you specified is not available in this script. You must choose a model part of this list :  ['VGG16', 'VGG19', 'ResNet50', 'Xception', 'InceptionV3', 'InceptionResNetV2', 'MobileNet', 'DenseNet121', 'DenseNet201', 'NASNetLarge', 'NASNetMobile']\n")
        sys.exit()

    return img_width, img_height, preprocessing


def testGenerator(modelName):

    global inputFolder

    img_width, img_height, preprocessing = modelSettings(modelName)

    test_datagen = ImageDataGenerator(preprocessing_function = preprocessing)

    test_generator = test_datagen.flow_from_directory(directory=r'./{}test'.format(inputFolder),
                                                        target_size=(img_width, img_height),
                                                        color_mode='rgb',
                                                        batch_size=batch_size,
                                                        class_mode='categorical',
                                                        shuffle=False)

    return test_generator


def load_model_fill_df(modelName, df):

    global inputFolder

    model = load_model(f'./Modeles/{modelName}.h5')
    test_generator = testGenerator(modelName, inputFolder)
    preds         = model.predict(test_generator)
    preds_classes = [i.argmax() for i in preds]
    y_true        = test_generator.classes

    acc_test = accuracy_score(y_true, preds_classes)
    prec_test = precision_score(y_true, preds_classes, average='weighted')
    rec_test = recall_score(y_true, preds_classes, average='weighted')
    f1_test = f1_score(y_true, preds_classes, average='weighted')
    params_number = model.count_params() 
    params_number_millions = round(params_number/1000000,1) #conversion in millions

    results = { 'modelName': modelName, 
                'accuracy': acc_test, 
                'precision': prec_test,
                'recall' : rec_test,
                'f1-score' : f1_test,
                'parameters_number (M)' : params_number_millions}

    df = df.append(results, ignore_index = True)

    return df


def add_execution_time(df):

    execution_time_dict = { 'VGG16' : 2663.36,
                            'VGG19' : 3123.99,
                            'ResNet50' : 2592,
                            'Xception' : 4824.38,
                            'InceptionV3' : 3538.52, 
                            'InceptionResNetV2' : 5430.35, 
                            'MobileNet' : 2554.67,
                            'DenseNet121' : 2878.75,
                            'DenseNet201' : 3466.23,
                            'NASNetLarge' : 9736.31,
                            'NASNetMobile' : 2744.64}

    #Conversion in minutes
    execution_time_dict = dict(map(lambda x: (x[0], round(x[1]/60, 2)), execution_time_dict.items()))

    df['execution_time (min)'] = df['modelName'].map(execution_time_dict)

    return df


if __name__ == "__main__":
    
    models_list = ["VGG16", "VGG19", "ResNet50", "Xception", "InceptionV3", "InceptionResNetV2", "MobileNet", "DenseNet121", "DenseNet201", "NASNetLarge", "NASNetMobile"]

    results_df = pd.DataFrame(columns=['modelName', 'accuracy', 'precision', 'recall', 'f1-score', 'parameters_number (M)'])

    for model in models_list:
        results_df = load_model_fill_df(model, results_df, inputFolder)

    results_df = results_df.sort_values(["accuracy"], ascending=False)
    results_df.reset_index(drop=True, inplace=True)

    #Add the execution time
    results_df = add_execution_time(results_df)

    results_df.to_csv("./Results/transfer_learning_compare_models.csv", index=False)

    print(f"\nDONE")
    