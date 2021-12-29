## Perceptron algorithm

A Perceptron is the most basic processing unit that we are going to find within a neural network, similar to a biological neuron. These Perceptron have input connections through which they receive external stimuli "the input values", with these values the Perceptron will perform an internal calculation and generate an output value.

i.e. A Perceptron is a mathematical function

what is this mathematical function about? : internally the Perceptron uses all the input values to make a weighted sum of them, the weight of each of the inputs is given by the weight assigned to each of the input connections, i.e. Each connection that reaches our Perceptron will have an associated value that will define with what intensity each input variable affects the Perceptron.

![](https://i.imgur.com/4PZ1jgH.png)



### Pseudocode

```pseudocode
dataset_train: import training data
dataset_test: import testing data

no_of_inputs = x
weigths = [w_1,...,w_i] 	# i = no_of_inputs 

(A) guess(inputs) #Activation score

    sum = np.dot(inputs, weigths) + bias
    if sum > 0
    	return 1
    else
    	return 0
	
(B) train()
	iterations = y
    learning_rate = z
    
    for iterations
    	for inputs, label in dataset_train
    		prediction = guess(inputs)
    		error = (label - prediction)
    		weights += learning_rate * error * inputs
    		bias += learning_rate * error 
    	

(C) test()
	correct = 0
	for inputs, label in dataset_test
		prediction = guess(inputs)
		if prediction == label[0]:
			correct +=1
	return correct / float(len(labels_arr))*100
```



## Implementation binary perceptron

When the algorithm was designed, the training data and testing data was given in an organised format, i.e. All instances of "class -1" precede instances of "class-2". 

This is bad for the generalisation of the perceptron, as it incentivizes the perceptron to only be effective with data alike to the organised data. by running the algorithm with this data organisation, we get the following results:

```
---------- Binary Approach (classX,classY) ----------

class 1 and class 2 --> Training accuracy: 99.69 | Testing accuracy: 100.00
class 2 and class 3 --> Training accuracy: 97.50 | Testing accuracy: 50.00
class 1 and class 3 --> Training accuracy: 99.69 | Testing accuracy: 100.00
```

We can clearly see that in the case of "class 2 and class 3" this is a problem as the training accuracy is quiet high giving a score of 97.5% however the testing accuracy is only  50%

by shuffling the dataset we make sure the data set is correctly trained "`np.random.shuffle(new_dataset)`", and this produces the following results:

```
---------- Binary Approach (classX,classY) ----------

class 1 and class 2 --> Training accuracy: 99.44 | Testing accuracy: 100.00
class 2 and class 3 --> Training accuracy: 77.25 | Testing accuracy: 60.00
class 1 and class 3 --> Training accuracy: 99.69 | Testing accuracy: 100.00
```



The pair for "class 2 and class 3" was the most difficult to separate, as it gave the least accuracy overall throughout all tests.

## Implementation multi-class classification

In order to implement this approach, an if statement was applied in the data_format function. (Cross Validated, n.d.)

If approach 1 was selected --> Binary Approach (classX,classY)

If approach 2 was selected --> 1-vs-rest Approach (classX vs rest)

```python
if approach == 1:
        # Filter: every time you find an array with class X & Y add it to new_dataset_tr
        for i in range(len(type_data)):
            if type_data[i,4:] == classX or type_data[i,4:] == classY:
                new_dataset.append(type_data[i])
    else:
        for i in range(len(type_data)):
            new_dataset.append(type_data[i])
```



Bellow we can find the accuracy report for this task:

```
---------- 1-vs-rest Approach (classX vs rest) ----------

class 1 vs rest --> Training accuracy: 99.71 | Testing accuracy: 100.00
class 2 vs rest --> Training accuracy: 59.96 | Testing accuracy: 63.33
class 3 vs rest --> Training accuracy: 85.00 | Testing accuracy: 90.00
```

## Implementation multi-class classification - regularised

While applying an l2 regularisation term of value 0.001, we get an improved accuracy compared to the results without regularisation, on the other hand values above these get worse as we test all of them.

#### λ= 0.001

```
---------- 1-vs-rest Approach (classX vs rest) - regularised ----------

class 1 vs rest --> Training accuracy: 99.71 | Testing accuracy: 100.00
class 2 vs rest --> Training accuracy: 67.25 | Testing accuracy: 70.00
class 3 vs rest --> Training accuracy: 90.67 | Testing accuracy: 100.00
```



#### λ= 0.01

```
---------- 1-vs-rest Approach (classX vs rest) - regularised ----------

class 1 vs rest --> Training accuracy: 99.71 | Testing accuracy: 100.00
class 2 vs rest --> Training accuracy: 50.08 | Testing accuracy: 53.33
class 3 vs rest --> Training accuracy: 93.92 | Testing accuracy: 96.67
```

#### λ= 0.1

```
---------- 1-vs-rest Approach (classX vs rest) - regularised ----------

class 1 vs rest --> Training accuracy: 99.75 | Testing accuracy: 100.00
class 2 vs rest --> Training accuracy: 44.33 | Testing accuracy: 36.67
class 3 vs rest --> Training accuracy: 66.62 | Testing accuracy: 66.67
```

#### λ= 1.0 

```
---------- 1-vs-rest Approach (classX vs rest) - regularised ----------

class 1 vs rest --> Training accuracy: 66.62 | Testing accuracy: 66.67
class 2 vs rest --> Training accuracy: 42.38 | Testing accuracy: 33.33
class 3 vs rest --> Training accuracy: 66.62 | Testing accuracy: 66.67
```

#### λ= 10.0

```
---------- 1-vs-rest Approach (classX vs rest) - regularised ----------

class 1 vs rest --> Training accuracy: 66.62 | Testing accuracy: 66.67
class 2 vs rest --> Training accuracy: 66.71 | Testing accuracy: 66.67
class 3 vs rest --> Training accuracy: 66.62 | Testing accuracy: 66.67
```

#### λ= 100.0

```
---------- 1-vs-rest Approach (classX vs rest) - regularised ----------

class 1 vs rest --> Training accuracy: 66.62 | Testing accuracy: 66.67
class 2 vs rest --> Training accuracy: 66.71 | Testing accuracy: 66.67
class 3 vs rest --> Training accuracy: 66.62 | Testing accuracy: 66.67
```
## Running

To run the code the following comand should be used:

- python MyPerceptron.py

By default the code will run using an l2 value of 0.001 but if user needs to test different values, the user can add arguments to this comand such as:

- python MyPerceptron.py 0.01 

