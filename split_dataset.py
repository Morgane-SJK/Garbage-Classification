"""
=============================================================================================================================================
Script which allows the creation of a folder with 3 subfolders (train test and validation) from the original Garbage Dataset folder of Kaggle
It can be downloaded here : https://www.kaggle.com/asdasdasasdas/garbage-classification
=============================================================================================================================================
"""

#IMPORTS
import getopt
import sys
import splitfolders
from os import listdir


#GLOBAL VARIABLES
folderKaggle = ""
outputFolder = "Garbage_train_test_val"
proportion_val = 0.15
proportion_test = 0.15
proportion_train = 0.70


def getArguments(): 
    
    global folderKaggle
    global proportion_val
    global proportion_test
    global proportion_train

    argsName = ["help", "folderKaggle","outputFolder", "proportion_val", "proportion_test", "proportion_train"]

    helpMessage = ''.join(["\nYou must use the script as follow :\n",
                           "python split_dataset.py --folderKaggle=\"\"\n\n",
                           "ALL OPTIONS:\n",
                           "--folderKaggle: path of the Garbage Classification Folder downloaded from Kaggle. It must contains one subfolder per class.\n",
                           "[OPTIONAL]\n",
                           "--proportion_val: proportion of images in the Validation set. Must be a float between 0 and 1. By default: 0.15 (so 15%).\n",
                           "--proportion_test: proportion of images in the Test set. Must be a float between 0 and 1. By default: 0.15 (so 15%).\n"])

    try:
        opts, args = getopt.getopt(sys.argv[1:],"",[arg+"=" if arg!="help" else arg for arg in argsName])
    
    except getopt.GetoptError as a:
        print(a)
        print("\nError Command Line Argument : python split_dataset.py --folderKaggle=\"\"\nFor Further information : python split_dataset.py --help\n")
        sys.exit()


    options = [opt for opt, arg in opts]

    #Check if the folderKaggle option is present
    if ("--folderKaggle" not in options) and ("--help" not in options):
        print("\nError Command Line Argument : python split_dataset.py --folderKaggle=\"\"\nFor Further information : python split_dataset.py --help\n")
        sys.exit()

    for opt, arg in opts:

        if   opt == "--help":
            print(helpMessage)
            sys.exit()

        elif opt == "--folderKaggle":
            folderKaggle=arg
            print("The path of folderKaggle is : ", folderKaggle)
            
        elif opt == "--outputFolder":
            outputFolder=arg
            print("The output folder is : ", outputFolder)

        elif opt == "--proportion_val":
            proportion_test=float(arg)
            
        elif opt == "--proportion_test":
            proportion_val=float(arg)

    proportion_train = 1 - proportion_val - proportion_test


if __name__ == "__main__":
    
    #We get the arguments given by the user
    getArguments()

    #Check if the path given for folderKaggle contains one subfolder per class
    if len(listdir(folderKaggle))<6:
        print("The path given for the Garbage dataset doesn't match the requirements. It must contains 6 subfolders : cardboard, glass, metal, paper, plastic and trash.")
        sys.exit()
    
    #We split the Garbage classification folder into 3 subfolders 
    splitfolders.ratio(folderKaggle, output=outputFolder, seed=777, ratio=(proportion_train, proportion_val, proportion_test))

    print(f"\nThe folder {outputFolder} has been successfully created.")
    