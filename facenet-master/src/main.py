import classify
import sys
import cv2
import preprocess
import time
import os

classifier = classify.Classify()
preprocessor = preprocess.PreProcessor()

camera = cv2.VideoCapture('Dataset/MPDS/War.mp4')
i = 0;
start = time.time()
while (camera.isOpened()):
    return_value, image = camera.read()
    cv2.imwrite('./Tempo/opencv'+str(i)+'.png', image)
    bb = (preprocessor.align('./Tempo/opencv'+str(i)+'.png'))
    
    if bb is not None:
        cv2.rectangle(image, (bb[0],bb[1]), (bb[2],bb[3]), (0, 255, 0), 5)
        name = classifier.predict('temp.png')
    else:
        cv2.imshow('frame',image)
        cv2.imwrite('./Pred/opencv'+str(i)+'.png', image)
        i+=1
        continue
        
    font = cv2.FONT_HERSHEY_SIMPLEX 
    cv2.putText(image, name, (50, 50), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow('frame',image)
    cv2.imwrite('./Pred/opencv'+str(i)+'.png', image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    i+=1
end_time = time.time()
print(i/(end_time-start))

camera.release()
cv2.destroyAllWindows()

