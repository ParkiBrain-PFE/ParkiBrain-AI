from datetime import datetime

def fetch_database(db_connection, number_plate):
  """"fetches the database for the number plate. if the number plate exists returns True otherwise returns False"""
  cur = db_connection.execute("SELECT * FROM cars_app_car WHERE number_plate=?", [number_plate]).fetchall()
  return len(cur) == 1
    
def update_last_enter(db_connection, number_plate):
  """"Updates the last_enter field in the database to indicate the last date the car used the parking"""
  db_connection.execute("UPDATE cars_app_car SET last_enter=? WHERE number_plate=?", [datetime.now(), number_plate])
  db_connection.commit()