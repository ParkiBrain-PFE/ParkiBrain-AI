from numpy import squeeze, expand_dims
from tflite_runtime.interpreter import Interpreter
from re import split

def load_model(path):
  """loads the given model to memory and return it's reference"""
  model = Interpreter(path)
  model.allocate_tensors()
  return model

def load_labels(path='labels.txt'):
  """Loads the labels file. Supports files with or without index numbers."""
  with open(path, 'r', encoding='utf-8') as f:
    lines = f.readlines()
    labels = {}
    for row_number, content in enumerate(lines):
      pair = split(r'[:\s]+', content.strip(), maxsplit=1)
      if len(pair) == 2 and pair[0].strip().isdigit():
        labels[int(pair[0])] = pair[1].strip()
      else:
        labels[row_number] = pair[0].strip()
  return labels

def set_input_tensor(interpreter, image):
  """Sets the input tensor."""
  tensor_index = interpreter.get_input_details()[0]['index']
  input_tensor = interpreter.tensor(tensor_index)()[0]
  input_tensor[:, :] = expand_dims((image-255)/255, axis=0)


def get_output_tensor(interpreter, index):
  """Returns the output tensor at the given index."""
  output_details = interpreter.get_output_details()[index]
  tensor = squeeze(interpreter.get_tensor(output_details['index']))
  return tensor


def detect_objects(interpreter, image, threshold):
  """Returns a list of detection results, each a dictionary of object info."""
  set_input_tensor(interpreter, image)
  interpreter.invoke()
  # Get all output details
  boxes = get_output_tensor(interpreter, 1)
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


def get_bounding_box(objects_array, camera_width=640, camera_height=480):
  """Gets the bounding box of the highest score object"""
  licence_plate = None
  sc = 0
  for obj in objects_array:
    if obj["score"] > sc:
      sc = obj["score"]
      licence_plate = obj["bounding_box"]

  if sc > 0:
    ymin, xmin, ymax, xmax = licence_plate
    xmin = int(max(1,xmin * camera_width))
    xmax = int(min(camera_width, xmax * camera_width))
    ymin = int(max(1, ymin * camera_height))
    ymax = int(min(camera_height, ymax * camera_height))
    licence_plate = (xmin, ymin, xmax, ymax) 
  return licence_plate
