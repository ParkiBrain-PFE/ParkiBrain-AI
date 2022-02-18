import cv2
from tflite_runtime.interpreter import Interpreter
from detection.detector import detect_objects

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

def get_bounding_box(objects_array):
  """Gets the bounding box of the highest score object"""
  licence_plate = None
  sc = 0
  for obj in objects_array:
    if obj["score"] > sc:
      sc = obj["score"]
      licence_plate = obj["bounding_box"]

  if sc > 0:
    ymin, xmin, ymax, xmax = licence_plate["bounding_box"]
    xmin = int(max(1,xmin * CAMERA_WIDTH))
    xmax = int(min(CAMERA_WIDTH, xmax * CAMERA_WIDTH))
    ymin = int(max(1, ymin * CAMERA_HEIGHT))
    ymax = int(min(CAMERA_HEIGHT, ymax * CAMERA_HEIGHT))
    licence_plate = (xmin, ymin, xmax, ymax) 
  return licence_plate

def main():
  interpreter = Interpreter('detect.tflite')
  interpreter.allocate_tensors()
  interpreter.get_input_details()[0]['shape']
  cap = cv2.VideoCapture(0)
  
  while cap.isOpened():
    _, frame = cap.read()
    img = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (320,320))
    objects_array = detect_objects(interpreter, img, 0.8)
    bounding_box = get_bounding_box(objects_array)
    if bounding_box:
      xmin, ymin, xmax, ymax = bounding_box
      cv2.rectangle(frame,(xmin, ymin),(xmax, ymax),(0,255,0),3)
      

    cv2.imshow('Camera', frame)
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
      cap.release()
      cv2.destroyAllWindows()

if __name__ == "__main__":
  main()