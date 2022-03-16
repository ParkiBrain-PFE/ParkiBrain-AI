from time import sleep
import cv2
from utils.allocate_resources import load_model
from utils.detect import detect_objects, get_bounding_box
from utils.plate_ocr import plate_to_string
from sqlite3 import connect
from utils.db_access import fetch_database, update_last_enter
 
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480



def main():
  detector = load_model(path='./models/detect.tflite')# load detection model
  recognizer = load_model(path='./models/best_recognizer.tflite')# load OCR model
  db_connection = connect("../db.sqlite3")
  cap = cv2.VideoCapture(0)
  
  while cap.isOpened():
    _, frame = cap.read()
    img = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (320,320))
    objects_array = detect_objects(detector, img, 0.65)
    bounding_box = get_bounding_box(objects_array, CAMERA_WIDTH, CAMERA_HEIGHT)
    if bounding_box:
      xmin, ymin, xmax, ymax = bounding_box
      plate = frame[ymin:ymax, xmin:xmax]
      plate = cv2.resize(plate, (320,320))
      content = detect_objects(recognizer, plate, 0.3)
      content = plate_to_string(content)      

      if content:
        # print(content)  
        is_authorized = fetch_database(db_connection, content)
        if is_authorized:
          update_last_enter(db_connection, content)
          print('ACCESS GRANTED')
          sleep(5)
          continue

    cv2.imshow('Camera', frame)
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
      db_connection.close()
      cap.release()
      cv2.destroyAllWindows()

if __name__ == "__main__":
  main()
