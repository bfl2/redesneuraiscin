import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

##Temos a class 1 como majoritaria, aproximadamente o dobro da class 0

def generate_class_split(class_dataset):
    dataset_length = class_dataset.shape[0]
    shuffled_dataset = class_dataset.sample(frac=1)
    train, testVal = train_test_split(shuffled_dataset, test_size=0.5)
    validate, test = train_test_split(testVal, test_size=0.5)
    print("sets: ",train.shape[0]/dataset_length, validate.shape[0]/dataset_length, test.shape[0]/dataset_length)
    return train, validate, test

def adjust_split(dataset):
    factor = 2 #para o dataset em questao
    if(factor==2):
        adjusted_split = dataset.append(dataset);
        adjusted_split = adjusted_split.sample(frac=1)

    return adjusted_split

def scaled_splits(dataset_class0):
    train, validate, test = generate_class_split(dataset_class0)
    trainw = adjust_split(train)
    validatew = adjust_split(validate)
    testw = adjust_split(test)

    return trainw, validatew, testw

def combine_split_class(split_class0, split_class1):
    #combine and shuffle dataset
    result = split_class0.append(split_class1).sample(frac=1)
    return result

def build_train_validation_test_sets(dataset_class0, dataset_class1):
    train0, validate0, test0 = scaled_splits(dataset_class0)
    train1, validate1, test1 = generate_class_split(dataset_class1)

    train = combine_split_class(train0, train1)
    validate = combine_split_class(validate0, validate1)
    test = combine_split_class(test0, test1)

    print("##Train, validation and test sets created##")
    print("sizes: ", train.shape[0], validate.shape[0], test.shape[0])
    return train, validate, test

def split_dataset():
    dataset = pd.read_csv("data/TRN", sep="\t")

    datapath = "data"
    #split in two classs
    dataset_class1 = dataset[dataset['IND_BOM_1_1']==1]
    dataset_class0 = dataset[dataset['IND_BOM_1_1']==0]
    train, validation, test = build_train_validation_test_sets(dataset_class0, dataset_class1)

    #print( "shape: ",dataset.shape[0])
    #print( "shape1: ",dataset_class1.shape)
    #print( "shape0: ",dataset_class0.shape)

    #write to file
    dataset_class1.to_csv(datapath+"/class/class1", sep='\t')
    dataset_class0.to_csv(datapath+"/class/class0", sep='\t')

    train.to_csv(datapath+"/train", sep='\t')
    validation.to_csv(datapath+"/validation", sep='\t')
    test.to_csv(datapath+"/test", sep='\t')

    #print(dataset_class1.head(5))
    #print(dataset_class0.head(5))

    print("###Finished splitting dataset###")
    return


split_dataset()
