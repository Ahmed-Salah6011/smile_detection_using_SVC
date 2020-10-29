# smile_detection_using_SVC
Training SVC model using sklearn to detect smile 

# Requirements
-sklearn
-pandas
-numpy
-opencv


# model.ipynb
This file contains the processing and loading of the data besides the model
## First Data is loaded 
The datasets.zip file contains the datasets folder
it contains 2 folders, one for train the other is for testing,
each one contains 2 folders one for smiling faces the other for normal faces
The data is loaded using cv2.imread() then turned into gray scale images then saved into a dataframe 


## Second the model
i use the SVC model supported by sklearn with kernel 'rbf'
the model is saved into smile_d.sav after training using joblib
You can find the pretrained model in smile_d.zip


# main.py
This file contains code loading the pretrained model and applying the model in a video stream
First we capture each frame from the video, then we detect the face in each fram using cv2.CascadeClassifier()
then the cropped face is converted to gray scale and is given to the model
