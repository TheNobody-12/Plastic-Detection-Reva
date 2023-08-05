import onnxruntime as ort
from flask import request, Flask, jsonify, render_template
from waitress import serve
from PIL import Image
import numpy as np
import exifread
import sqlite3
from fractions import Fraction
import os
import time
import uuid

# import functions from the AtlanticWarriors_function.py file
from AtlanticWarriors_function import detect_objects_on_image, get_image_geolocation, store_data_in_database, fetch_lat_lon_from_db, fetch_img_names_and_counts_from_db
# use geopy to get location from lat and lon
from geopy.geocoders import Nominatim
geolocator = Nominatim(user_agent="object-detection-app")

app = Flask(__name__)
db_path = "object_detection_data.db"


def main():
    # serve(app, host='0.0.0.0', port=8080)
    app.run(debug=True)


@app.route("/")
def root():
    """
    Site main page handler function.
    :return: Content of index.html file
    """

    # with open("templates/index.html") as file:
    #     return file.read()
    return render_template("index.html")

@app.route("/predict")
def predict():
    return render_template("predict.html")

@app.route("/detect", methods=["POST"])
def detect():
    """
    Handler of /detect POST endpoint
    Receives uploaded file with a name "image_file", passes it
    through YOLOv8 object detection network and returns an array
    of bounding boxes along with geolocation from the image metadata.
    The data is also stored in a SQLite database.
    :return: a JSON array of objects containing bounding boxes in format [[x1, y1, x2, y2, object_type, probability], ...]
    """
    buf = request.files["image_file"]
    filename = buf.filename
    print(filename)
    boxes = detect_objects_on_image(buf.stream)

    # Get geolocation from the image metadata
    geolocation = get_image_geolocation(buf)

    # Store the data in the SQLite database
    store_data_in_database(filename,boxes, geolocation)

    # return jsonify(boxes, geolocation)
    return jsonify(boxes)

# create different route for the database
@app.route("/database", methods=["GET"])
def database():
    """
    Handler of /database GET endpoint
    Retrieves data from the SQLite database and returns it as a JSON array
    :return: a JSON array of objects containing bounding boxes in format [[x1, y1, x2, y2, object_type, probability], ...]
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM object_detection_data")
    rows = cursor.fetchall()
    conn.close()

    return jsonify(rows)

@app.route("/get_lat_lon/<filename>")
def get_lat_lon(filename):
    print("Received filename:", filename)
    lat_lon = fetch_lat_lon_from_db(filename)
    print("Latitude and Longitude:", lat_lon)
    return jsonify(lat_lon)

@app.route("/get_location/<lat>/<lon>")
def get_location(lat, lon):
    print("Received lat and lon:", lat, lon)
    location = geolocator.reverse(f"{lat}, {lon}", exactly_one=True)
    print("Location:", location)

    if location:
        location_data = {
            "address": location.address,
            "country": location.raw.get("address", {}).get("country"),
            "postcode": location.raw.get("address", {}).get("postcode")
        }
    else:
        location_data = {
            "address": "Location data not found",
            "country": "Unknown",
            "postcode": "Unknown"
        }

    return jsonify(location_data)



@app.route("/db_data")
def db_data():
    # Fetch data from the database
    filenames_data, lat_lon_data = fetch_lat_lon_from_db()

    db = {
        "filenames": filenames_data,
        "lat_lon": lat_lon_data
    }

    return jsonify(db)

@app.route("/db")
def db():
    return render_template("db.html")



@app.route("/linechart")
def linechart():
    # Fetch data from the database (for example, using fetch_img_names_and_counts_from_db() function)
    img_names, no_of_plastic = fetch_img_names_and_counts_from_db()

    return jsonify(img_names, no_of_plastic)

@app.route("/linecharts.html")
def linecharts():
    return render_template("linecharts.html")



if __name__ == "__main__":
    main()
