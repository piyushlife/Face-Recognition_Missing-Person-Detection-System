# Face-Recognition_Missing-Person-Detection-System
Missing Person Detection System or, MPDS is a solution or a system aims primarily at finding a person which goes “missing” as reported, with as high accuracy as possible using the latest state of the art machine learning and deep learning technologies.


This piece of application makes use of facenet model developed by davidsandberg for face detection and recognition. Please refer this link for more info : https://github.com/davidsandberg/facenet

The models can be downloaded from [here](https://drive.google.com/open?id=1bXEJmjmd750F53DsDv9B2vc1thbxlJrE).
You can also get the link to download latest model from [davidsandberg](https://github.com/davidsandberg/facenet) repository too. Please make sure to change the value of "FACENET_MODEL_PATH" as is mentioned below.

Create a folder named "facenet" inside "facenet-master/Models". Copy all the downloaded model files and paste it inside the newly created folder.

#### Note: Please be sure to make changes to the value of the variables "CLASSIFIER_PATH", "EMOTION_MODEL_PATH" and "FACENET_MODEL_PATH" in faceRec.py file inside src folder according to the changes you made.

This application is used to recognise a person on which our model is trained as well as to get his/her emotion.

With this application, we have provided a "mpds.pkl" file which is currently trained on 3 different classes: Me and my two other colleague. You can straightaway run this application with the below mentioned steps:
	1. Get inside facenet-master folder.
	2. Open Terminal or cmd on the same location.
	3. Fire this command: `"python src/faceRec.py"` or `"python src/faceRec.py --path <path_of_video/image_file>"`
	4. Press `"q"` or `"Esc"` to close the pop up screen.
	
Alternatively, you can delete the pickle file and train your own classes and then follow the steps above again to view the result.