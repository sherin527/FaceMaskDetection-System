import plaidml.keras
plaidml.keras.install_backend()
from  keras.models import load_model
import cv2
import numpy as np
import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()
#tf.compat.v1.disable_resource_variables()


model = load_model('face_mask_detection_alert_system6.h5')

face_det_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

vid_capture = cv2.VideoCapture(0)

text_dict={0:'Mask_on', 1:'No_mask'}
rect_color_dict = {0:(0,0,255),1:(0,0,255)}



while(True):
    ret,img = vid_capture.read()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    face = face_det_classifier.detectMultiScale(gray,1.3,5)
    
    for(x,y,w,h) in face:
        face_img = gray[y:y+w,x:x+w]
        resized_img = cv2.resize(face_img,(112,112))
        normalized_img = resized_img/255.0
        reshaped_img = np.reshape(normalized_img,(1,112,112,1))
        result=model.predict(reshaped_img)

        label = np.argmax(result,axis=1)[0]

        cv2.rectangle(img,(x,y),(x+w,y+h), rect_color_dict[label],2)
        cv2.rectangle(img,(x,y-40),(x+w,y),rect_color_dict[label],-1)
        cv2.putText(img,text_dict[label], (x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,0),2)
        
        





        
        
        
        #cv2.rectangle(img,(x,y),(x+w,y+h), (255,0,0),4)
        
        #cv2.rectangle(img,(x,y-40),(x+w,y),rect_color_dict[label],-1)
        
        
        
        
    cv2.imshow('LIVE',img)
    key=cv2.waitKey(30) & 0xff
    if(key==27):
        break
vid_capture.release()
cv2.destroyAllWindows()

