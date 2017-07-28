#**Behavioral Cloning** 


** Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/nvidia-architecture.png "Model Visualization"
[image2]: ./examples/normal_left_camera.png "Normal Image (Left Camera)"
[image3]: ./examples/normal_center_camera.png "Normal Image (Center Camera)"
[image4]: ./examples/normal_right_camera.png "Normal Image (Right Camera)"
[image5]: ./examples/recovery_left_camera.png "Recovery Image (Left Camera)"
[image6]: ./examples/recovery_center_camera.png "Recovery Image (Center Camera)"
[image7]: ./examples/recovery_right_camera.png "Recovery Image (Right Camera)"
[image8]: ./examples/distribution.png "Distribution"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

I use the same model in the Nvdia paper: https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/

#### 2. Attempts to reduce overfitting in the model

No overfitting problem encountered due to the huge amount of data collected.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 104).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road, driving reversely.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

####1. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded total of 12 laps (6 clockwise and 6 counterclockwise) on track one using center lane driving. Here is an example image of center lane driving (all three cameras):

![normal driving - left camera][image2] ![normal driving - center camera][image3] ![normal driving - right camera][image4]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to drive back to center when it's off the center. These images show what a recovery looks like (all three cameras):

![recovery - left camera][image5] ![recovery - center camera][image6] ![recovery - right camera][image7]

I also recorded additional two laps (1 clockwise and 1 counterclockwise) on curve only to make sure the car learns to make sharp turns.

I didn't flip the image as I garthered enough data both clockwise and counter-clockwise, which makes training faster. In order to avoid the bias from too much data points which has ~0 steering angles, I only kept 20% of data points of its kind. In addition, I use images from all three cameras with some adjustment on the steering angles (+0.25 for left camera image and -0.25 for right camera image).

Here is the distribution after data collection and processing step:
![data distribution][image8]

After all the above steps, I had **109305** number of data points.


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting.


