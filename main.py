import re
import cv2
from detection.utils import detect_objects, get_bounding_box, load_model, load_labels

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
labels = load_labels(path='labels.txt')


def plate_to_string(plate)->str:
  if len(plate) < 4:
    return None
  content = ''
  sorted_list = sorted(plate, key=lambda k: k["bounding_box"][1])
  for obj in sorted_list:
    content += str(labels[int(obj['class_id'])])
  return content


def main():
  detector = load_model('./models/detect.tflite')
  recognizer = load_model('./models/recognize.tflite')

  cap = cv2.VideoCapture(0)
  
  while cap.isOpened():
    _, frame = cap.read()
    img = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (320,320))
    objects_array = detect_objects(detector, img, 0.8)
    bounding_box = get_bounding_box(objects_array, CAMERA_WIDTH, CAMERA_HEIGHT)
    if bounding_box:
      xmin, ymin, xmax, ymax = bounding_box
      plate = frame[ymin:ymax, xmin:xmax]
      plate = cv2.resize(plate, (320,320))
      content = detect_objects(recognizer, plate, 0.5)
      content = plate_to_string(content)      

      if content:
        print(content)      

    cv2.imshow('Camera', frame)
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
      cap.release()
      cv2.destroyAllWindows()

if __name__ == "__main__":
  main()



# def process_img(img):
#   """
#   Process image by removing noise and transforming pixels above 100 to white and others below to black.
#   """
#   img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#   img = cv2.GaussianBlur(img,(5,5),0)
#   img = cv2.threshold(img, 100, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY)[1] 
#   return img