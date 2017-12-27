# **Traffic Sign Recognition** 

The goals / steps of this Udacity project were the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

Below you can find the summary of the results. 

The Jupyter file containing the project code is [sdc_traffic_sign_classifier](https://github.com/frtunikj/sdc_traffic_sign_classifier/blob/master/Traffic_Sign_Classifier.ipynb).

[//]: # (Image References)

[image0]: ./examples/output_8_0.png 
[image1]: ./examples/output_9_0.png 
[image2]: ./examples/output_9_1.png 
[image3]: ./examples/output_9_2.png
[image4]: ./examples/output_14_1.png
[image5]: ./examples/Lenet5.png
[image6]: ./examples/LenetPaper.png
[image7]: ./examples/output_31_1.png
[image8]: ./examples/output_38_1.png
[image9]: ./examples/output_43_1.png
[image10]: ./examples/output_44_1.png

---

### Data Set Summary & Exploration

To see a statistics summary of the traffic signs data set, the pandas library was used. The data set was already split into training, validation and test set. Below is a short summary of the data set.

* The size of training set is 34799
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3) i.e. 32x32 pixels in RGB.
* The number of unique classes/labels in the data set is 43

#### Exploratory visualization of the dataset

To get an overview of the data set two helper functions for showing images and plotting histogram were written. 

A set of 15 random images from the training data set is shown below:

![alt text][image0]

In order to see the distribution of data among the training, validation and test set, a histogram per traffic sign class was used.

![alt text][image1]

![alt text][image2]

![alt text][image3]

From the plots above one can notice that there's a strong imbalance among the classes i.e. some classes are relatively over-represented, while some others are much less common. However, the data distribution is almost the same between training, testing and the validation set which is good. 

One potential improvement of the data set will be to generate additional data for the traffic sign classes that e.g. have less than 1000 samples in the training set. This could be done by rotating images, projection, adding noise to an image etc.

### Design and Test a Model Architecture

#### 1. Preprocessing of data

The paper from [Sermanet, LeCun](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf) that also targets traffic sign classification suggests performing data preprocessing in order to improve the accuracy of the network. The following steps were suggested and also performed below:

1. Convert each image is converted from RGB to YUV color space and use only the Y channel. The formula for calculating the Y channel is: Y = 0.299R + 0.587G + 0.114B.

2. Adjust the contrast of each image by means of histogram equalization. This step is required to mitigate the numerous situation in which the image contrast is really poor.

3. Center each image is on zero mean and divide it for its standard deviation. This feature scaling is known to have beneficial effects on the gradient descent performed by the optimizer (see lectures http://cs231n.github.io).

To perform the above steps a function preprocessImages was created. A set of 15 random processed images is shown below:

![alt text][image4]

#### 2. Design model architectures

Two types of convolutional networks were considered, evaluated and implemented:

A. The original [LeNet-5](http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf)

![alt text][image5]

The architecture of the original LeNet-5 network was adopted as following:

* The input was adapted to the preprocessed 32x32x1 images (not 32x32x3 as in the original). 
* Dropout was performed on the fc1 and fc2 (fully conected) layers.
* The output layer/logits was adopted to 43 since there are 43 unique classes in the set (not 10 as in the original network for digits classification).


B. The [2-Stage CovNet](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf)

![alt text][image6]

The final model was the 2-Stage CovNet and it consists of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 out 				|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 10x10x16 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x6 out 				|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 1x1x400 	|
| RELU					|												|
| Flatten    			| Layers (1x1x400 -> 400) and (5x5x16 -> 400)	|
| Concatenate  			| Layers from previos steps to single 800 layer	|
| Dropout				|												|
| Fully connected layer | input 800, output 43        					|

The final 2-Stage CovNet architecture is a shallow network. The first two layers are convolutional, while the third (convolutional) and last are fully connected. As described in the [paper](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf) the output of both the second and the third convolution layers are concatenated and fed to the following fully connected layer. This is done in order to provide the fully-connected layer visual patterns at both different levels of abstraction. The fully connected layer then maps the prediction into one of the 43 classes.

Both networks performed similar i.e. validation accuracy of approximately 95% and test accuracy of approximately 93%. Detailed explanation of the training procedure is given in the sections below.

#### 3. Hyperparameter description

For the training the Adam optimizer was used. The Adam optimizer is a good choice to avoid the patient search of the right parameters for SGD. Batchsize was set to 128 and the number of epochs was set to 30. In order to avoid overfitting, dropout with a probability of 0.5 was employed. The other hyperparameter were set as following: mu = 0, sigma = 0.1, learning rate = 0.001, dropout keep probability: 0.5.

#### 4. Training process description

The final 2-Stage CovNet model results were:
* validation set accuracy of 94.3% 
* test set accuracy of 92.6%

For developing and training the model, I started with pre-defined architectures (LeNet-5 and the 2-Stage CovNet). The tweaking process from there on was more of trial and error. The starting validation accuracy that was measured was approximately 91%. In order to get a validation accuracy of more that 0.93, the following steps were performed (on both arcihtectures):

* First the data set was preprocessed as already described above (RGB to YUV - only the Y channel, and normalizing the input w.r.t. to the mean and standard deviation). With this the validation accuracy went up to approximately 95%.
* In order to avoid overfitting, the dropout technique with a probability of 0.5. was applied. One can use the L1 or L2 regularization for this purpose as well. Important to note - in order to see how the model performs on the validation set, no dropouts are made i.e. dropout probability is set to 1.0.
* Since there was a variation in the validation accuracy, the learning rate was reduced to 0.001.

Changing the batch size to 64 or 256 did not affect the validation accuracy at all. Also varying the epoch size to 60 or reducing the learning rate changed only slightly the accuracy. 

### Test a Model on New Images

#### 1. Selection of 5 test images

In the next step 5 images were selected from the web for testing purposes. 

![alt text][image7] 

The second (go straight or left) and the third (dangerous curve on the right) image were selected for testing purposes of the model because these two classes only had 180 samples in the training set (compared to other classes with more that 700 samples). The first image (speed limit 50Km/h) was selected in order to see how the model performs on images/classes that are well represented in the training set but has a bit rotated view angle. The last two images (priority road and yield) were selected as good examples (no rotation, no noise, enough training examples).

#### 2. Model prediction of 5 test images 

The results of the prediction are listed in the table below:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed limit 50  		| Speed limit 50     							| 
| Go straight or left  	| Speed limit 30								|
| Dangerous curve right	| Dangerous curve right							|
| Priority road	      	| Priority road					 				|
| Yield     			| Yield              							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. One can notice the performance drop w.r.t. the test set (approximately 13%). However, one has to keep in mind that 5 images are too few to be of any statistical significance.

#### 3. Softmaxscore visualization of 5 test images

The figure below displays the top 3 prediction probabilities of the model for each of the 5 test images.

![alt text][image8]  

One can see that the model is quite sure of which class the third and the fourth traffic sign belong. The model is slightly confused on the first and the fifth image. Regarding the first image, the model might be a bit unsecure because the image is a bit rotated. For the second image (go straight or left) the model's prediction is wrong (as might have been expected), which might be a result of not enough training samples. 

### 4. Neural Network Visualization

While neural networks can be a great learning device they are often referred to as a black box. We can understand what the weights of a neural network look like better by plotting their feature maps. After successfully training your neural network you can see what it's feature maps look like by plotting the output of the network's weight layers in response to a test stimuli image. From these plotted feature maps, it's possible to see what characteristics of an image the network finds interesting. For a sign, maybe the inner network feature maps react with high activation to the sign's boundary outline or to the contrast in the sign's painted symbol.

The figures below show the activations of the first two feature map layers for one example input (fourth image from above), i.e. priority road sign. One can see esspecially from the output of the first feature layer that the network learns on its own to detect the boundaries (in this case four of them) of the traffic sign. Moreover, one can notice also the contrast in the feature map. Unfortunately from the second deature map layer I can not see what the network exactly learned. 

![alt text][image9] 

![alt text][image10] 

### References

Further useful readings:

* http://cs231n.github.io/ (Modile 1 and Module 2 lectures) 
* https://en.wikipedia.org/wiki/YUV
* http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf
* https://navoshta.com/traffic-signs-classification/ 