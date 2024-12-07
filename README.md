# Logistic Regression for Binary Classification
Logistic regression is one of the most important analytic tools in the social and natural sciences. It is used to classify an observation into one of two classes or into one of many classes. In this article, we use the MNIST dataset to classify two classes.
## Methodology
1. Data Preprocessing
2. Forward Pass
3. backpropagation
4. learning parameter update
## Data Preprocessing
The MNIST dataset is downloaded, and the input data is created, including ```x_train, y_train, x_test, and y_test``` , with the selected classes being the numbers 8 and 9. The shape of ```x_train``` is ```[11800, 28, 28]``` , indicating there are ```11,800``` input images, each of size ```28x28``` pixels. The shape of y_train is ```[11800, 1]```, which represents the corresponding class labels for each input image in ```x_train``` and This structure is similar for ```x_test``` and ```y_test``` as well. ```x_train``` and ```x_test``` are flattened along dimensions 1 and 2. The new shape of ```x_train``` is ```[11800, 784]```, meaning there are ```11,800``` input images, each represented by ```784``` features. This is similar to ```x_test```. The number ```9``` is mapped to ```1``` and the number ```8``` to ```0``` in ```y_train``` and ```y_test``` to represent the positive and negative classes, respectively, as we are performing binary classification. ```x_train``` and ```y_train``` are split into an ```80-20``` ratio for validation (```x_val and y_val```).
## Forward Pass
### 1. Linear Model
Logistic regression addresses binary classification by learning (weight and bias) from the training data. Each weight w<sub>i</sub> is a real number associated with an input feature x<sub>i</sub>. The weight w<sub>i</sub> indicates the importance of that input feature in the classification decision. It can be positive, suggesting that the instance being classified likely belongs to the positive class, or negative, indicating that the instance is more likely to belong to the negative class. The bias term, also called the intercept, is another real number that’s added to the weighted inputs.<br><br>
For binary classification, the linear model is given by:<br>
```z=xw+b```<br><br>
For ```784``` input features, the weight vector ```w``` will have a shape of ```[784, 1]```, and the bias ```b``` will have a shape of ```[1, 1]```. The matrix multiplication of ```x``` and ```w``` results in a shape of ```[number_of_data, 1]```. The bias ```b``` is broadcasted along the rows to match the shape of ```xw```, and the addition operation is performed. Finally, ```z``` will have a shape of ```[number_of_data, 1]```. Each element in ```z``` is the result of the dot product of the input vector and the weight vector, plus the bias.
### 2. Sigmoid
### 3. Binary Cross Entropy Loss
## Backpropagation
## Update Learning Parameter
