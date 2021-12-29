import sys
import numpy as np


lambda_=0.001

if len(sys.argv) == 2:
    lambda_=float(sys.argv[1])

# ---------- Load training data ----------

# Convert function - Converts "class-x" string to x integer
convert_func = lambda x: float(x.strip(b"class-")) 

#import dataset using the numpy library and parsing the convertfunc
dataset_train = np.genfromtxt('CA1data/train.data',delimiter=',',converters={4: convert_func})


# ---------- Load testing data ----------

# Convert function - Converts "class-x" string to x integer
convert_func = lambda x: float(x.strip(b"class-")) 

#import dataset using the numpy library and parsing the convertfunc
dataset_test = np.genfromtxt('CA1data/test.data',delimiter=',',converters={4: convert_func})



# ---------- Format data ----------

def format_data(classX=1,classY=2,type_data = dataset_train,approach = 1):
    new_dataset= []
    if approach == 1:
        # Filter: every time you find an array with class X & Y add it to new_dataset_tr
        for i in range(len(type_data)):
            if type_data[i,4:] == classX or type_data[i,4:] == classY:
                new_dataset.append(type_data[i])
    else:
        for i in range(len(type_data)):
            new_dataset.append(type_data[i])

    
    new_dataset = np.asarray(new_dataset)

    # Shuffle new dataset
    np.random.seed(42)
    np.random.shuffle(new_dataset)

    # Divide dataset into inputs and labels 
    input_arr = new_dataset[:,:4]
    labels_arr = new_dataset[:,4:]

    #print (labels_arr)
    # Set binary labels (0,1)
    for i in range(len(labels_arr)):
        if labels_arr [i] == classX:
            labels_arr [i] = 0
        else:
            labels_arr [i] = 1
    
    return (input_arr, labels_arr)


# ---------- Guess ----------

def guess(inputs): 

    sum = np.dot(inputs, weights[1:]) + weights[0]
    #activation function
    if sum > 0:
        activation = 1
    else:
        activation = 0
    return activation


# ---------- Train ----------
def train(classX=1,classY=2,approach=1):
    iterations = 20
    learning_rate = 0.01
    
    input_arr, labels_arr = format_data(classX,classY,dataset_train,approach)
    correct = 0
    actual = 0
    for _ in range(iterations):
        for inputs, label in zip(input_arr, labels_arr):
            prediction = guess(inputs)

            weights[1:] += learning_rate *((label - prediction) * inputs)
            weights[0] += learning_rate * (label - prediction)
            actual +=1
            if prediction == label[0]:
                correct +=1
            #print (prediction, label)
    return correct / float(actual)*100
        
    #print (weights)



# ---------- Test ----------

def test(classX=1,classY=2,approach=1):
    #print (weights)
    input_arr, labels_arr = format_data(classX,classY,dataset_test,approach)

    correct = 0
    for inputs, label in zip(input_arr, labels_arr):
        prediction = guess(inputs)
        if prediction == label[0]:
            correct +=1
        #print (prediction, label)
    return correct / float(len(labels_arr))*100
    


# ---------- regularisation training ----------Lambda
def regularisation(classX=1,classY=2,approach=1):
    iterations = 20
    learning_rate = 1
    


    input_arr, labels_arr = format_data(classX,classY,dataset_train,approach)
    correct = 0
    actual = 0
    for _ in range(iterations):
        for inputs, label in zip(input_arr, labels_arr):
            prediction = guess(inputs)
            weights[1:] += (label - prediction) * inputs + (2*lambda_*weights[1:])
            weights[0] += learning_rate * (label - prediction)
            actual +=1
            if prediction == label[0]:
                correct +=1
            #print (prediction, label)
    return correct / float(actual)*100
      
# ---------- initialise weights variables globally ----------
no_of_inputs = 4
weights = np.zeros(no_of_inputs + 1)



# ---------- Main Binary Approach (classX,classY) ----------


# class 1 and class 2


classX = 1
classY = 2

accuracy_train = train(classX,classY)
accuracy_test = test(classX,classY)
weights = np.zeros(no_of_inputs + 1) #empty weigths 
print ("\n---------- Binary Approach (classX,classY) ----------\n")
print("class 1 and class 2 --> Training accuracy:","%.2f" % accuracy_train, "| Testing accuracy:","%.2f" %  accuracy_test )

# class 2 and class 3

classX = 2
classY = 3

accuracy_train = train(classX,classY)
accuracy_test = test(classX,classY)
weights = np.zeros(no_of_inputs + 1) #empty weigths 

print("class 2 and class 3 --> Training accuracy:","%.2f" % accuracy_train, "| Testing accuracy:","%.2f" %  accuracy_test )

# class 1 and class 3

classX = 1
classY = 3

accuracy_train = train(classX,classY)
accuracy_test = test(classX,classY)
weights = np.zeros(no_of_inputs + 1) #empty weigths 

print("class 1 and class 3 --> Training accuracy:","%.2f" % accuracy_train, "| Testing accuracy:","%.2f" %  accuracy_test )


# ---------- Main 1-vs-rest Approach (classX vs rest) ----------



# class 1 vs rest

classX = 1
classY = 0

accuracy_train = train(classX,classY,2)
accuracy_test = test(classX,classY,2)
weights = np.zeros(no_of_inputs + 1) #empty weigths 

print ("\n---------- 1-vs-rest Approach (classX vs rest) ----------\n")
print("class 1 vs rest --> Training accuracy:","%.2f" % accuracy_train, "| Testing accuracy:","%.2f" %  accuracy_test )

# class 2 vs rest

classX = 2
classY = 0

accuracy_train = train(classX,classY,2)
accuracy_test = test(classX,classY,2)
weights = np.zeros(no_of_inputs + 1) #empty weigths 

print("class 2 vs rest --> Training accuracy:","%.2f" % accuracy_train, "| Testing accuracy:","%.2f" %  accuracy_test )

# class 3 vs rest

classX = 3
classY = 0

accuracy_train = train(classX,classY,2)
accuracy_test = test(classX,classY,2)
weights = np.zeros(no_of_inputs + 1) #empty weigths 

print("class 3 vs rest --> Training accuracy:","%.2f" % accuracy_train, "| Testing accuracy:","%.2f" %  accuracy_test )


# ---------- 1-vs-rest Approach (classX vs rest) - regularised (l2) ----------



# class 1 vs rest

classX = 1
classY = 0

accuracy_train = regularisation(classX,classY,2)
accuracy_test = test(classX,classY,2)
weights = np.zeros(no_of_inputs + 1) #empty weigths 

print ("\n---------- 1-vs-rest Approach (classX vs rest) - regularised ----------\n")
print("class 1 vs rest --> Training accuracy:","%.2f" % accuracy_train, "| Testing accuracy:","%.2f" %  accuracy_test )

# class 2 vs rest

classX = 2
classY = 0

accuracy_train = regularisation(classX,classY,2)
accuracy_test = test(classX,classY,2)
weights = np.zeros(no_of_inputs + 1) #empty weigths 

print("class 2 vs rest --> Training accuracy:","%.2f" % accuracy_train, "| Testing accuracy:","%.2f" %  accuracy_test )

# class 3 vs rest

classX = 3
classY = 0

accuracy_train = regularisation(classX,classY,2)
accuracy_test = test(classX,classY,2)
weights = np.zeros(no_of_inputs + 1) #empty weigths 

print("class 3 vs rest --> Training accuracy:","%.2f" % accuracy_train, "| Testing accuracy:","%.2f" %  accuracy_test )
