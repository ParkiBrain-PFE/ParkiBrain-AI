from time import sleep
import re
import cv2
from tflite_runtime.interpreter import Interpreter
import numpy as np
from easyocr import Reader

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

def set_input_tensor(interpreter, image):
  """Sets the input tensor."""
  tensor_index = interpreter.get_input_details()[0]['index']
  input_tensor = interpreter.tensor(tensor_index)()[0]
  input_tensor[:, :] = np.expand_dims((image-255)/255, axis=0)


def get_output_tensor(interpreter, index):
  """Returns the output tensor at the given index."""
  output_details = interpreter.get_output_details()[index]
  tensor = np.squeeze(interpreter.get_tensor(output_details['index']))
  return tensor


def detect_objects(interpreter, image, threshold):
  """Returns a list of detection results, each a dictionary of object info."""
  set_input_tensor(interpreter, image)
  interpreter.invoke()
  # Get all output details
  boxes =get_output_tensor(interpreter, 1)
  classes = get_output_tensor(interpreter, 3)
  scores = get_output_tensor(interpreter, 0)
  count = int(get_output_tensor(interpreter, 2))

  results = []
  for i in range(count):
    if scores[i] >= threshold:
      result = {
        'bounding_box': boxes[i],
        'class_id': classes[i],
        'score': scores[i]
      }
      results.append(result)
  return results

def main():
  #reader = Reader(["en", "ar"]) #load esay ocr model in RAM
  interpreter = Interpreter(model_path="detect.tflite") #
  interpreter.allocate_tensors()
  cap = cv2.VideoCapture(0) #start the video capture from the camera
  print ("ok")
  while cap.isOpened(): #an infinite loop that will try to detect the licence plate and do bunch of treatments
    _, frame = cap.read() #load the current frame
    img = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (320,320)) #resizing the frame and chaging its color(BRG -->RGB) convention so the costum model will be able to take the frame as input
    objects_array = detect_objects(interpreter, img, 0.8) #using the costum model to detect objects in the frame and returning the location of the objects
    
    """licence_plate = None
    sc = 0
    for obj in objects_array: #loop that retain the location of the object that is the most likely to be our deseired object
      if obj["score"] > sc:
        sc = obj["score"]
        licence_plate = obj["bounding_box"]

    if sc > 0:  #if an object is detected
      ymin, xmin, ymax, xmax = licence_plate
      xmin = int(max(1,xmin * CAMERA_WIDTH))
      xmax = int(min(CAMERA_WIDTH, xmax * CAMERA_WIDTH))
      ymin = int(max(1, ymin * CAMERA_HEIGHT))
      ymax = int(min(CAMERA_HEIGHT, ymax * CAMERA_HEIGHT))
      
      #cv2.rectangle(frame,(xmin, ymin),(xmax, ymax),(0,255,0),3)#draw a rectangle around the object
      #cv2.imshow("Hello", frame[ymin:ymax, xmin:xmax])
      #croped_img = frame[ymin:ymax, xmin:xmax]  #crop the img so it contains only the object
      cv2.imshow('Camera', frame) #show the image of the frame in the screan
      #cap.release() #stop the camera from capturing the flow (it's more of a way to reduce resource usage)
      #res = reader.readtext(croped_img, detail=0) #read the text inside the croped image using esay ocr
      #print(res)  #print the result of deading process
      #sleep(2)  #here will be our treatments
      #cap = cv2.VideoCapture(0) #start the camera

    else:"""
    cv2.imshow('Camera', frame)
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
      cap.release()
      cv2.destroyAllWindows()


if __name__ == "__main__":
  main()