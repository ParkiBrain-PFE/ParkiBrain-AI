from numpy import squeeze, expand_dims
from tflite_runtime.interpreter import Interpreter

def load_model(path):
  """loads the given model to memory and return it's reference"""
  model = Interpreter(path)
  model.allocate_tensors()
  return model

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