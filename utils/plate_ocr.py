from .constants import LABELS

def plate_to_string(plate)-> str:
  if len(plate) < 4:
    return None
  content = ''
  sorted_list = sorted(plate, key=lambda k: k["bounding_box"][1])
  for obj in sorted_list:
    content += str(LABELS[int(obj['class_id'])])
  return content