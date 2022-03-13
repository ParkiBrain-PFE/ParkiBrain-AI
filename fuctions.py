def licence_plate_filter(objects_array):
    licence_plate_coordinates = None
    sc = 0
    for obj in objects_array: #loop that retain the location of the object that is the most likely to be our deseired object
        if obj["score"] > sc:
            sc = obj["score"]
            licence_plate_coordinates = obj["bounding_box"]
    return licence_plate_coordinates,sc