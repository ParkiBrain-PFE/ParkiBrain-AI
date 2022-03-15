from utils.allocate_resources import get_output_tensor, set_input_tensor


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
  if len(objects_array) == 0:
    return None
  licence_plate = max(objects_array, key=lambda k: k["score"])# get the highest score licence plate
  ymin, xmin, ymax, xmax = licence_plate["bounding_box"]
  xmin = int(max(1,xmin * camera_width))
  xmax = int(min(camera_width, xmax * camera_width))
  ymin = int(max(1, ymin * camera_height))
  ymax = int(min(camera_height, ymax * camera_height))
  licence_plate = (xmin, ymin, xmax, ymax) 
  return licence_plate
