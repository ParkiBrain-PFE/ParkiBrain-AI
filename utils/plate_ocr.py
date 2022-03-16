LABELS = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "a", "b", "d", "h", "w", "waw"]

def plate_to_string(plate)-> str:
  if len(plate) < 4:
    return None
  content = ''
  sorted_list = sorted(plate, key=lambda k: k["bounding_box"][1]) #sort characters based on x_min
  for obj in sorted_list:
    content += str(LABELS[int(obj['class_id'])])
  return content