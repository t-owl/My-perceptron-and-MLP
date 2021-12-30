# Perceptron
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

```python MyPerceptron.py```

By default the code will run using an l2 value of 0.001 but if user needs to test different values, the user can add arguments to this comand such as:

```python MyPerceptron.py 0.01```

# Multilayer perceptron (MLP)
In the previous task we worked with a single layer perceptron, which help us understand how by applying linear regression we could classify binary data such as AND, and OR gates. To overcome more complex classification problems we would then use a combination of perceptron layers, problems such as XOR gates or even problems with more complexity.

There are two ways of putting neurons together, one is to put them in the same layer (same column), as we can see bellow neurons on the same layer will receive the output from the previous layer and they will then output their result to the next layer. By placing layers next to each other we can achieve hierarchical knowledge, in other words the first layers of the network will hold basic information (e.g. borders in a picture, corners in a picture, etc), where as the last layers will have a more abstract knowledge combining previous layers (e.g. cars in a picture, trees in a picture, etc). 

![](https://i.imgur.com/VvERANv.png)



There is a problem with this approach, if we were just to add many perceptrons together, we would end up with the equivalent of 1 perceptron, as all a perceptron does is to solve a linear regression problem, i.e. adding many linear regression problems is equivalent to just one. However to overcome this we choose to activation functions that output a non linear result (Sigmoid function, ReLU function, etc).

Once we've build our network we need to train it, but unlike the single perceptron we have many weights to adjust, depending on the size of the network adjusting every weight would not be viable, so that is where we used the backpropagation algorithm. I order to mitigate this problem we walk backwards, instead of walking from the input to the output which then produces a loss signal, we start from the loss signal and walk backwards, this is possible because in a neural network the mistake in previous layers directly affects later layers.


### Implementation 

The data set used in this MLP was extracted from Sklearn, we used two circles as for the classification task the created dataset is shown below.

![](https://i.imgur.com/CKTdCjx.png)

Once the data is created we can now build our neural network, to start with we need our layer layout, this is implemented by the `neural_layer():` class:

- In this class we need the number of connections that go as input from previous layers `n_con`
- We also have to initiate the number of neurons that are currently in this layer `n_neuron`
- The bias and weight are also initialised
- Finally we pass the activation function `act_f` (Sigmoid function, ReLU function, etc).

Now we define our activation functions, each function is in charge of generating non linear outputs which makes possible the combination of many perceptrons, each function is stated as a formula but a derivative is also needed for the backpropagation algorithm, (Codesansar, 2019):


- **Sigmoid function:** 

  - Mathematical function

  
    <img src="https://render.githubusercontent.com/render/math?math=f(x) = \sigma(x) = \frac{1}{1+e^{-x}}#gh-light-mode-only"><img src="https://render.githubusercontent.com/render/math?math=\color{White}f(x) = \sigma(x) = \frac{1}{1+e^{-x}}#gh-dark-mode-only">
   

    ![](https://i.imgur.com/F1gJOqI.png)

  - Derivative
    
    <img src="https://render.githubusercontent.com/render/math?math=f'(x) = \sigma(x) ( 1 - \sigma(x) )#gh-light-mode-only"><img src="https://render.githubusercontent.com/render/math?math=\color{White}f'(x) = \sigma(x) ( 1 - \sigma(x) )#gh-dark-mode-only">
    
 
    ![](https://i.imgur.com/6gzVEX1.png)


- **Tangent Hyperbolic Function:** 

  - Mathematical function
    
    <img src="https://render.githubusercontent.com/render/math?math=f(x) = \frac{e^{x} - e^{-x}}{e^{x} + e^{-x}}#gh-light-mode-only"><img src="https://render.githubusercontent.com/render/math?math=\color{White}f(x) = \frac{e^{x} - e^{-x}}{e^{x} + e^{-x}}#gh-dark-mode-only">
    
    
    ![](https://i.imgur.com/jeWKdv7.png)

  - Derivative
    
    <img src="https://render.githubusercontent.com/render/math?math=f'(x) = ( 1 - g(x)^{2} )#gh-light-mode-only"><img src="https://render.githubusercontent.com/render/math?math=\color{White}f'(x) = ( 1 - g(x)^{2} )#gh-dark-mode-only">
    
    ![](https://i.imgur.com/UhoS4e7.png)


- **Rectified Linear Unit (RELU) Function** 

  - Mathematical function
    
    <img src="https://render.githubusercontent.com/render/math?math=f(x) = max(0,x)#gh-light-mode-only"><img src="https://render.githubusercontent.com/render/math?math=\color{White}f(x) = max(0,x)#gh-dark-mode-only">
    
    
    ![](https://i.imgur.com/4PH5Ogg.png)

  - Derivative
    
    <img src="https://render.githubusercontent.com/render/math?math=f(x) = \begin{cases} \text{1, x>0} \\ \text{0, otherwise} \end{cases}#gh-light-mode-only"><img src="https://render.githubusercontent.com/render/math?math=\color{White}f(x) = \begin{cases} \text{1, x>0} \\ \text{0, otherwise} \end{cases}#gh-dark-mode-only">
    
    
    ![](https://i.imgur.com/2uJP640.png)


Once the activation function is created, we defined a function which is in charge of creating  our neural network `create_nn()`, our neural network will be defined by our `topology` variable which will hold the structure of the neural network.

After the creation of our network we will train our network, training has 3 essential elements, Forward pass,  Backward pass (Backpropagation) and Gradient descent (mlnotebook.github.io, 2017) 

Forward pass takes an input and an output (desired output), and processes them by using activation functions and basic formulas to generate an output. during the first stages of the training the output generated will be a random value, but as we continue the training the initial value will change into a closer value to the desire output.(Skalski, 2018)

Backward pass (Backpropagation), once forward pass generates an output, this would then be compared against the desired output (real value) by using the cost function, this tells us how different are the results to each other which results in an error value. The backpropagation algorithm takes the error value generated as well as the  derivatives of the activation functions, to calculate the partial derivatives.

1. **Computation of the error of the last layer**


	<img src="https://render.githubusercontent.com/render/math?math=\delta^{L}=\frac{\partial C}{\partial a^{L}} \cdot \frac{\partial a^{L}}{\partial z^{L}}#gh-light-mode-only"><img src="https://render.githubusercontent.com/render/math?math=\color{White}\delta^{L}=\frac{\partial C}{\partial a^{L}} \cdot \frac{\partial a^{L}}{\partial z^{L}}#gh-dark-mode-only">



2. **Backpropagate the error to the previous layer**


	<img src="https://render.githubusercontent.com/render/math?math=\delta^{l-1}=W^{l} \delta^{l} \cdot \frac{\partial a^{l-1}}{\partial z^{l-1}}#gh-light-mode-only"><img src="https://render.githubusercontent.com/render/math?math=\color{White}\delta^{l-1}=W^{l} \delta^{l} \cdot \frac{\partial a^{l-1}}{\partial z^{l-1}}#gh-dark-mode-only">



3. **Calculate the derivatives of the layer using the error**


	<img src="https://render.githubusercontent.com/render/math?math=\frac{\partial \alpha}{\partial b^{l-1}}=\delta^{l-1} \quad \frac{\partial C}{\partial w^{l-1}}=\delta^{l-1} a^{l-2}#gh-light-mode-only"><img src="https://render.githubusercontent.com/render/math?math=\color{White}\frac{\partial \alpha}{\partial b^{l-1}}=\delta^{l-1} \quad \frac{\partial C}{\partial w^{l-1}}=\delta^{l-1} a^{l-2}#gh-dark-mode-only">


Gradient descent takes the pre-calculated partial derivatives, to optimise the cost function, i.e. it will train our network. (Santana Vega, 2018)

### Output/ experiments

As part of the experiment I was able to produce some interesting results on the input data, by trying different activation functions 

To further illustrate the outcome a graph was produced showing the classification as well as the error graph following the gradient descent.



#### Sigmoid Classification

The Sigmoid activation function had a great performance over the input data, by classifying the dataset in the least amount of iterations (1000) with a learning rate equal to 0.05, Sigmoid tends to perform better in binary classification, consequently working well with the problem presented 



![](https://i.imgur.com/EFKa2eg.png)

![](https://i.imgur.com/S3LpwJV.png)

#### Tanh or Hyperbolic Classification

As the algorithm was build around the Sigmoid function there was a similar performance when it came to Tanh optimising the cost function on the same number of iterations as Sigmoid (1000), on the other hand a smaller learning rate performed better on this case learning rate = 0.001

![](https://i.imgur.com/lHrOZQf.png)

![](https://i.imgur.com/R9NTJfx.png)



#### Rectified Linear Unit (RELU) Clasification

ReLU reflected the worst performance over all, this would be due to the nature of ReLU, the neural network algorithm uses the same activation function across all its layers, where as ReLU  should only be used within the hidden layers. 

As part of the results, in most of the runs of the algorithm, neurons died through the training which resulted in a non correct output.

*"Another problem with ReLu is that some gradients can be fragile during training and can die. It can cause a weight update which will makes it never activate on any data point again. Simply saying that ReLu could result in Dead Neurons"* (Omkar, 2019)

In the cases where the neurons did not died the algorithm used the following values: learning rate=0.005, iterations = 20000. and the graph produced where the following:

![](https://i.imgur.com/kzkiR1n.png)

![](https://i.imgur.com/SH5pevk.png)


