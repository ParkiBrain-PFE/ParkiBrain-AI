from time import sleep
import re
import cv2
from tflite_runtime.interpreter import Interpreter
import pytesseract

import TensorFlowFunctions as TFfunc
import fuctions as fct




CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480



def main():
  interpreter = Interpreter(model_path="Converted_model.tflite") #load the model
  interpreter.allocate_tensors()
  cap = cv2.VideoCapture(0) #start the video capture from the camera
  while cap.isOpened(): #an infinite loop that will try to detect the licence plate and do bunch of treatments
    _, frame = cap.read() #load the current frame
    img = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (320,320)) #resizing the frame and chaging its color(BRG -->RGB) convention so the costum model will be able to take the frame as input
    objects_array = TFfunc.detect_objects(interpreter, img, 0.7) #using the costum model to detect objects in the frame and returning the location of the objects
    
    licence_plate,score = fct.licence_plate_filter(objects_array)

    if score != 0:  #if an object is detected
      ymin, xmin, ymax, xmax = licence_plate
      xmin = int(max(1,xmin * CAMERA_WIDTH))
      xmax = int(min(CAMERA_WIDTH, xmax * CAMERA_WIDTH))
      ymin = int(max(1, ymin * CAMERA_HEIGHT))
      ymax = int(min(CAMERA_HEIGHT, ymax * CAMERA_HEIGHT))
      
      cv2.rectangle(frame,(xmin, ymin),(xmax, ymax),(0,255,0),3)#draw a rectangle around the object
      cv2.imshow("Hello", frame[ymin:ymax, xmin:xmax])
      croped_img = frame[ymin:ymax, xmin:xmax]  #crop the img so it contains only the object
      cv2.imshow('Camera', frame) #show the image of the frame in the screan
      cap.release() #stop the camera from capturing the flow (it's more of a way to reduce resource usage)


      print("eng"+pytesseract.image_to_string(croped_img,lang='eng',config="-c tessedit_char_whitelist=0123456789|/\w"))
      print("ara",pytesseract.image_to_string(img,lang='ara',config="-c tessedit_char_whitelist=أبدهـط"))

      sleep(2)

      cap = cv2.VideoCapture(0) #start the camera



    else:
      cv2.imshow('Camera', frame)
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
      cap.release()
      cv2.destroyAllWindows()


if __name__ == "__main__":
  main()