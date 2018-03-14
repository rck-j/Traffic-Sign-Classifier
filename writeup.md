
### Question 1 

_Describe how you preprocessed the data. Why did you choose that technique?_

**Answer**

I tried a few different things and nothing seemed to improve my outcomes.  So, I went with converting the data set to YUV and normalizing the Y channel with contrast limited adaptive histogram equalization (CLAHE).  CLAHE was suggested to me in my first submission.  This gave me the most success when during training.  

### Question 2

_Describe how you set up the training, validation and testing data for your model. **Optional**: If you generated additional data, how did you generate the data? Why did you generate the data? What are the differences in the new dataset (with generated data) from the original dataset?_

**Answer**

I ran the image preprocessing on all of the data.  I didn’t create extra data.  In my first submission I split the data, but we now have a training, validation, and testing sets.


### Question 3

_What does your final architecture look like? (Type of model, layers, sizes, connectivity, etc.)  For reference on how to build a deep neural network using TensorFlow, see [Deep Neural Network in TensorFlow
](https://classroom.udacity.com/nanodegrees/nd013/parts/fbf77062-5703-404e-b60c-95b78b2f3f9e/modules/6df7ae49-c61c-4bb2-a23e-6527e69209ec/lessons/b516a270-8600-4f93-a0a3-20dfeabe5da6/concepts/83a3a2a2-a9bd-4b7b-95b0-eb924ab14432) from the classroom._


**Answer**
I started with LeNet, and was going to submit LeNet because I wasn’t able to get better results with other architectures.  I finally tried to implement VGG.  VGG starts with larger images, so it’s not the entire architecture.  I cut off the last set of convolutional layers.  The following is an image I modeled my architecture off of.
[image1]: ./vgg.jpg

![alt text][image1]

I went as far as the as the first 512 set of layers.  The pooling layer boiled the dimensions down to 2x2.  Running another set of convolutions didn’t seem to make since.  I also changed the fully connected layer from 4096 to 301.  The 4096 wasn't a good fit for the number of classes we have with our data.

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 3x3     	| 1x1 stride, same padding, outputs 32x32x64 	|
| RELU					| Activation									|
| Convolution 3x3       | 1x1 stride, same padding, outputs 32x32x64    |
| RELU                  | Activation                                    |
| Max pooling	      	| 2x2 stride,  outputs 16x16x64 				|
| Convolution 3x3	    | 1x1 stride, same padding, outputs 16x16x128  	|
| RELU          		| Activation        							|
| Convolution 3x3       | 1x1 stride, same padding, outputs 16x16x128   |
| RELU                  | Activation                                    |
| Max pooling	      	| 2x2 stride,  outputs 8x8x128 				    |
| Convolution 3x3       | 1x1 stride, same padding, outputs 8x8x256     |
| RELU                  | Activation                                    |
| Convolution 3x3       | 1x1 stride, same padding, outputs 8x8x256     |
| RELU                  | Activation                                    |
| Convolution 3x3       | 1x1 stride, same padding, outputs 8x8x256     |
| RELU                  | Activation                                    |
| Max pooling	      	| 2x2 stride,  outputs 4x4x256 	     			|
| Convolution 3x3       | 1x1 stride, same padding, outputs 4x4x512     |
| RELU                  | Activation                                    |
| Convolution 3x3       | 1x1 stride, same padding, outputs 4x4x512     |
| RELU                  | Activation                                    |
| Convolution 3x3       | 1x1 stride, same padding, outputs 4x4x512     |
| RELU                  | Activation                                    |
| Max pooling	      	| 2x2 stride,  outputs 2x2x512 	     			|
| Flatten               | Outputs 2048                                  |
| Fully Connected       | Inputs 2048 Outputs 301                       |
| Fully Connected       | Inputs 301 Outputs 301                        |
| Fully Connected       | Inputs 301  Outputs 43                        |


### Question 4

_How did you train your model? (Type of optimizer, batch size, epochs, hyperparameters, etc.)_

**Answer**

The biggest bang for my buck were the learning rate, the number of features in the fuly connected layers and the number of epochs.  I started with 10 epochs in the beginning because I was training on my mac, but it wasn’t enough time for the model to run, so I adjusted to 60 and moved to AWS.  This brought my score out of the trenches.   The next improvement was reducing the number of features for the fully connected layers.  I was trying to implement VGG and they used 4096.  This was to high for the number of classes we have. I reduced it to 301 and that got me to the 90s.  Then I played with the learning rate. This got me to the mid 90s.  

### Question 5


_What approach did you take in coming up with a solution to this problem? It may have been a process of trial and error, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think this is suitable for the current problem._

**Answer**

Lots of trial and error.  I finally tried to implement a known architecture.  I went with VGG because it was the easiest to understand.  The diagram is pretty straight forward.  

### Question 6

_Choose five candidate images of traffic signs and provide them in the report. Are there any particular qualities of the image(s) that might make classification difficult? It could be helpful to plot the images in the notebook._



[image2]: ./images/images.jpg
[image3]: ./images/images-2.jpg
[image4]: ./images/images-3.jpg
[image5]: ./images/images-4.jpg
[image6]: ./images/images-6.jpg

![alt text][image2] 
![alt text][image3]
![alt text][image4]
![alt text][image5]
![alt text][image6]

**Answer**

These are the images I pulled from the web.  And the following are the resized I tested on.


[image7]: ./images/small_images.jpg
[image8]: ./images/small_images-2.jpg
[image9]: ./images/small_images-3.jpg
[image10]: ./images/small_images-4.jpg
[image11]: ./images/small_images-6.jpg

![alt text][image7] 
![alt text][image8]
![alt text][image9]
![alt text][image10]
![alt text][image11]

I was only able to classify one of the images.  I think the problem was is in the resizing.  I cropped the images to a square dimension that was a multiple of 32 to try and resize them as nicely as possible.  I don’t think it worked.  

### Question 7

_Is your model able to perform equally well on captured pictures when compared to testing on the dataset? The simplest way to do this check the accuracy of the predictions. For example, if the model predicted 1 out of 5 signs correctly, it's 20% accurate._

_**NOTE:** You could check the accuracy manually by using `signnames.csv` (same directory). This file has a mapping from the class id (0-42) to the corresponding sign name. So, you could take the class id the model outputs, lookup the name in `signnames.csv` and see if it matches the sign from the image._


**Answer**

I only correclty classified 1 image for an accuracy of 20%. 

### Question 8

*Use the model's softmax probabilities to visualize the **certainty** of its predictions, [`tf.nn.top_k`](https://www.tensorflow.org/versions/r0.12/api_docs/python/nn.html#top_k) could prove helpful here. Which predictions is the model certain of? Uncertain? If the model was incorrect in its initial prediction, does the correct prediction appear in the top k? (k should be 5 at most)*

`tf.nn.top_k` will return the values and indices (class ids) of the top k predictions. So if k=3, for each sign, it'll return the 3 largest probabilities (out of a possible 43) and the correspoding class ids.

Take this numpy array as an example:

```
# (5, 6) array
a = np.array([[ 0.24879643,  0.07032244,  0.12641572,  0.34763842,  0.07893497,
         0.12789202],
       [ 0.28086119,  0.27569815,  0.08594638,  0.0178669 ,  0.18063401,
         0.15899337],
       [ 0.26076848,  0.23664738,  0.08020603,  0.07001922,  0.1134371 ,
         0.23892179],
       [ 0.11943333,  0.29198961,  0.02605103,  0.26234032,  0.1351348 ,
         0.16505091],
       [ 0.09561176,  0.34396535,  0.0643941 ,  0.16240774,  0.24206137,
         0.09155967]])
```

Running it through `sess.run(tf.nn.top_k(tf.constant(a), k=3))` produces:

```
TopKV2(values=array([[ 0.34763842,  0.24879643,  0.12789202],
       [ 0.28086119,  0.27569815,  0.18063401],
       [ 0.26076848,  0.23892179,  0.23664738],
       [ 0.29198961,  0.26234032,  0.16505091],
       [ 0.34396535,  0.24206137,  0.16240774]]), indices=array([[3, 0, 5],
       [0, 1, 4],
       [0, 5, 1],
       [1, 3, 5],
       [1, 4, 3]], dtype=int32))
```

Looking just at the first row we get `[ 0.34763842,  0.24879643,  0.12789202]`, you can confirm these are the 3 largest probabilities in `a`. You'll also notice `[3, 0, 5]` are the corresponding indices.

**Answer**

```
TopKV2(values=array([[  9.99986291e-01,   1.27114326e-05,   7.39129689e-07, 2.12067334e-07,   6.38713038e-09],
                     [  1.00000000e+00,   2.08287970e-11,   1.37219871e-14, 1.57252395e-16,   1.08844965e-16],
                     [  9.98359263e-01,   1.64066639e-03,   2.70867559e-08, 2.39759046e-11,   4.35794196e-12],
                     [  8.26857328e-01,   1.72925115e-01,   1.17233321e-04, 6.58388599e-05,   3.40926417e-05],
                     [  8.07663739e-01,   1.54890388e-01,   2.71607712e-02, 7.44388951e-03,   2.53533362e-03]], dtype=float32), 
indices=array([[13, 10,  9, 12, 42],
               [17,  4, 41, 25, 10],
               [12, 13, 14, 18, 25],
               [13,  4, 38,  8,  2],
               [25, 23, 41, 10, 32]], dtype=int32))
               
   my_labels = [17, 11, 12,  2, 15]
   
```
               
These were my results and the correct answers.  I was only able to classify sign 2, but this looks like it was only .0000349% sure that it was 2. Hmmm.

