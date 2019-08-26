from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import argparse
import facenet
import os
import sys
import math
import pickle
import align.detect_face
import numpy as np
import cv2
import collections
from sklearn.svm import SVC
from thread import WebcamVideoStream as WBS

from keras.models import load_model
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import os
import cv2
import numpy as np
from os import listdir
from os.path import isfile, join
from keras.preprocessing.image import img_to_array

def logic(frame):
    bounding_boxes, _ = align.detect_face.detect_face(frame, MINSIZE, pnet, rnet, onet, THRESHOLD, FACTOR)
    faces_found = bounding_boxes.shape[0]
    print("Faces Found: ", faces_found)

    try:
        if faces_found > 0:
            temp_dict = {}
            det = bounding_boxes[:, 0:4]
            bb = np.zeros((faces_found, 4), dtype=np.int32)
            for i in range(faces_found):
                bb[i][0] = det[i][0]
                bb[i][1] = det[i][1]
                bb[i][2] = det[i][2]
                bb[i][3] = det[i][3]

                top_down_diff = abs(int(bb[i][0])-int(bb[i][2]))
                right_left_diff = abs(int(bb[i][1])-int(bb[i][3]))

                if top_down_diff < 35 and right_left_diff < 35:
                    continue

                cropped = frame[bb[i][1]:bb[i][3], bb[i][0]:bb[i][2], :]

                gray = cv2.cvtColor(cropped,cv2.COLOR_BGR2GRAY)
                roi_gray = cv2.resize(gray, (48, 48), interpolation = cv2.INTER_AREA)
                roi = roi_gray.astype("float") / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)

                scaled = cv2.resize(cropped, (INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE),  interpolation=cv2.INTER_CUBIC)
                scaled = facenet.prewhiten(scaled)
                scaled_reshape = scaled.reshape(-1, INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE, 3)
                feed_dict = {images_placeholder: scaled_reshape, phase_train_placeholder: False}

                emb_array = sess.run(embeddings, feed_dict=feed_dict)
                predictions = model.predict_proba(emb_array)
                best_class_indices = np.argmax(predictions, axis=1)
                best_class_probabilities = predictions [  np.arange(len(best_class_indices)), best_class_indices]
                best_name = class_names[best_class_indices[0]]

                preds = emo_classifier.predict(roi)[0]
                label = class_labels[preds.argmax()]

                print("Name: {}, Probability: {}".format(best_name, best_class_probabilities))

                if best_class_probabilities < 0.7:
                    cv2.rectangle(frame, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), (0, 255, 0), 2)
                    continue
                else:
                    cv2.rectangle(frame, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), (0, 255, 0), 2)
                    if i == 0:
                        
                        temp_dict[best_name] = [bb[i], best_class_probabilities, label]
                    else:
                        if best_name in temp_dict:
                            
                            if temp_dict[best_name][1] < best_class_probabilities:
                                temp_dict[best_name] = [bb[i], best_class_probabilities, label]
                        else:
                            temp_dict[best_name] = [bb[i],best_class_probabilities, label]

            
            print("Temp Dict: ",temp_dict)
            for name, value in temp_dict.items():
                text_x = value[0][0]
                text_y = value[0][3] + 20
                emo = value[2]
                cv2.putText(frame, name, (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX_SMALL,1, (255, 255, 255), thickness=1, lineType=2)
                cv2.putText(frame, str(round(value[1][0], 3)), (text_x, text_y+17), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), thickness=1, lineType=2)
                cv2.putText(frame, emo, (text_x, text_y+34), cv2.FONT_HERSHEY_COMPLEX_SMALL,1, (255, 255, 255), thickness=1, lineType=2)

                person_detected[best_name] += 1


    except Exception as e:
        print(e)
        pass

    cv2.imshow('Face Recognition',cv2.cvtColor(frame,cv2.COLOR_RGB2BGR))
    return cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)



parser = argparse.ArgumentParser()
parser.add_argument('--path', help = 'Video path', default = 0)
args = parser.parse_args()

MINSIZE = 20
THRESHOLD = [0.6, 0.7, 0.7]
FACTOR = 0.709
IMAGE_SIZE = 182
INPUT_IMAGE_SIZE = 160
CLASSIFIER_PATH = 'Models/MPDS/mpds.pkl'
EMOTION_MODEL_PATH = 'Models/MPDS/model_72per_all_200_pics.hdf5'
VIDEO_PATH = args.path
FACENET_MODEL_PATH = 'Models/facenet/20180402-114759.pb'

class_labels = {'angry': 0, 'disgust': 1, 'happy': 2, 'neutral': 3, 'sad': 4, 'surprised': 5}
class_labels = {v: k for k, v in class_labels.items()}

emo_classifier = load_model(EMOTION_MODEL_PATH)
print("The Emotion Model was Successfully loaded")

with open(CLASSIFIER_PATH, 'rb') as file:
   model, class_names = pickle.load(file)
print("The Custom Classifier was Successfully loaded")

with tf.Graph().as_default():
    gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.6)
    sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options, log_device_placement = False))
    
    with sess.as_default():
        print('Loading feature extraction model')
        facenet.load_model(FACENET_MODEL_PATH)


    images_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("input:0")
    embeddings = tf.compat.v1.get_default_graph().get_tensor_by_name("embeddings:0")
    phase_train_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("phase_train:0")
    embedding_size = embeddings.get_shape()[1]


    pnet, rnet, onet = align.detect_face.create_mtcnn(sess, "./src/align")

people_detected = set()
person_detected = collections.Counter()

if VIDEO_PATH == 0:
    web = WBS().start()
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter("./Dataset/result/Web.avi", fourcc, 20.0, (640,480))

    while True:
        frame = web.read()
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

        #Resize frame into 1024*768

        if frame.shape[0] >768 and frame.shape[1]>1024:
            frame = cv2.resize(frame,(1024,768))  

        img = logic(frame)
        out.write(img)
        
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break

    web.stop() 
    out.release()
    cv2.destroyAllWindows()

elif os.path.splitext(VIDEO_PATH)[1] in ['.mp4', '.flv', '.avi', '.mpg', '.mkv']:
    cap = cv2.VideoCapture(VIDEO_PATH)
    filename = VIDEO_PATH.split("/")[-1].split(".")[0]
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter("./Dataset/result/{}.avi".format(filename), fourcc, 20.0, (640,352))
    try:
        while(cap.isOpened()):
            
            ret, frame = cap.read()
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            
            #Resize frame into 1024*768

            if frame.shape[0] >768 and frame.shape[1]>1024:
                frame = cv2.resize(frame,(1024,768)) 

            img = logic(frame)
            out.write(img)

            if cv2.waitKey(1) & 0xFF == ord('q'): 
                break
        
        cap.release()
        out.release()
        cv2.destroyAllWindows()
    
    except Exception as e:
        print(e)

elif os.path.splitext(VIDEO_PATH)[1] in ['.jpg', '.png', '.jpeg']:
    cap = cv2.imread(VIDEO_PATH,1)
    frame = cv2.cvtColor(cap,cv2.COLOR_BGR2RGB)

    # Resize frame into 1024*768
    
    if frame.shape[0] >768 and frame.shape[1]>1024:
            frame = cv2.resize(frame,(1024,768))

    img = logic(frame)
    filename = VIDEO_PATH.split("/")[-1]
    cv2.imwrite("./Dataset/result/{}".format(filename),img)

    k = cv2.waitKey(0)
    if k == 27:         # wait for ESC key to exit
        cv2.destroyAllWindows()
    elif k == ord('s'): # wait for 's' key to exit
       cv2.destroyAllWindows()
