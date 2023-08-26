import onnxruntime as ort
from flask import request, Flask, jsonify, render_template
from waitress import serve
from PIL import Image
import pandas as pd
import plotly.express as px 
import pandas as pd
import numpy as np
import exifread
import sqlite3
from fractions import Fraction
import os
import time
import uuid

# use geopy to get location from lat and lon
from geopy.geocoders import Nominatim
geolocator = Nominatim(user_agent="object-detection-app")

"""

"""
def detect_objects_on_image(stream):
    input, img_width, img_height = prepare_input(stream)
    output = run_model(input)
    return process_output(output, img_width, img_height)



def prepare_input(buf):
    """
    Function used to convert input image to tensor,
    required as an input to YOLOv8 object detection
    network.
    :param buf: Uploaded file input stream
    :return: Numpy array in a shape (3,width,height) where 3 is number of color channels
    """
    img = Image.open(buf)
    img_width, img_height = img.size
    img = img.resize((2176, 2176))
    img = img.convert("RGB")
    input = np.array(img) / 255.0
    input = input.transpose(2, 0, 1)
    input = input.reshape(1, 3, 2176, 2176)
    return input.astype(np.float32), img_width, img_height


def run_model(input):
    """
    Function used to pass provided input tensor to
    YOLOv8 neural network and return result
    :param input: Numpy array in a shape (3,width,height)
    :return: Raw output of YOLOv8 network as an array of shape (1,84,8400)
    """
    model = ort.InferenceSession("Phase2_TeamAtlanticModel.onnx")
    outputs = model.run(["output0"], {"images":input})
    return outputs[0]


def process_output(output, img_width, img_height):
    """
    Function used to convert RAW output from YOLOv8 to an array
    of detected objects. Each object contain the bounding box of
    this object, the type of object and the probability
    :param output: Raw output of YOLOv8 network which is an array of shape (1,84,8400)
    :param img_width: The width of original image
    :param img_height: The height of original image
    :return: Array of detected objects in a format [[x1,y1,x2,y2,object_type,probability],..]
    """
    output = output[0].astype(float)
    output = output.transpose()

    boxes = []
    for row in output:
        prob = row[4:].max()
        if prob < 0.2:
            continue
        class_id = row[4:].argmax()
        label = yolo_classes[class_id]
        xc, yc, w, h = row[:4]
        x1 = (xc - w/2) / 2176 * img_width
        y1 = (yc - h/2) / 2176 * img_height
        x2 = (xc + w/2) / 2176 * img_width
        y2 = (yc + h/2) / 2176 * img_height
        boxes.append([x1, y1, x2, y2, label, prob])

    boxes.sort(key=lambda x: x[5], reverse=True)
    result = []
    while len(boxes) > 0:
        result.append(boxes[0])
        boxes = [box for box in boxes if iou(box, boxes[0]) < 0.3] 

    return result


def iou(box1,box2):
    """
    Function calculates "Intersection-over-union" coefficient for specified two boxes
    https://pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/.
    :param box1: First box in format: [x1,y1,x2,y2,object_class,probability]
    :param box2: Second box in format: [x1,y1,x2,y2,object_class,probability]
    :return: Intersection over union ratio as a float number
    """
    return intersection(box1,box2)/union(box1,box2)


def union(box1,box2):
    """
    Function calculates union area of two boxes
    :param box1: First box in format [x1,y1,x2,y2,object_class,probability]
    :param box2: Second box in format [x1,y1,x2,y2,object_class,probability]
    :return: Area of the boxes union as a float number
    """
    box1_x1,box1_y1,box1_x2,box1_y2 = box1[:4]
    box2_x1,box2_y1,box2_x2,box2_y2 = box2[:4]
    box1_area = (box1_x2-box1_x1)*(box1_y2-box1_y1)
    box2_area = (box2_x2-box2_x1)*(box2_y2-box2_y1)
    return box1_area + box2_area - intersection(box1,box2)


def intersection(box1,box2):
    """
    Function calculates intersection area of two boxes
    :param box1: First box in format [x1,y1,x2,y2,object_class,probability]
    :param box2: Second box in format [x1,y1,x2,y2,object_class,probability]
    :return: Area of intersection of the boxes as a float number
    """
    box1_x1,box1_y1,box1_x2,box1_y2 = box1[:4]
    box2_x1,box2_y1,box2_x2,box2_y2 = box2[:4]
    x1 = max(box1_x1,box2_x1)
    y1 = max(box1_y1,box2_y1)
    x2 = min(box1_x2,box2_x2)
    y2 = min(box1_y2,box2_y2)
    return (x2-x1)*(y2-y1)


# Array of YOLOv8 class labels
yolo_classes = ["0"]



def get_image_geolocation(file):
    # Function to extract geolocation from image metadata using exifread library
    file.seek(0)
    tags = exifread.process_file(file)

    latitude_ref = tags.get("GPS GPSLatitudeRef")
    latitude = tags.get("GPS GPSLatitude")
    longitude_ref = tags.get("GPS GPSLongitudeRef")
    longitude = tags.get("GPS GPSLongitude")

    if latitude_ref and latitude and longitude_ref and longitude:
        latitude = parse_exif_gps_value(latitude)
        longitude = parse_exif_gps_value(longitude)

        # Convert the latitude and longitude from degrees, minutes, seconds to decimal degrees
        latitude_dec = convert_dms_to_dd(latitude)
        longitude_dec = convert_dms_to_dd(longitude)

        # Adjust the sign of latitude and longitude based on their reference
        if latitude_ref.values == "S":
            latitude_dec = -latitude_dec
        if longitude_ref.values == "W":
            longitude_dec = -longitude_dec

        return {"latitude": latitude_dec, "longitude": longitude_dec}
    else:
        raise ValueError("Geolocation data not found in image metadata.")


def parse_exif_gps_value(value):
    # Helper function to parse EXIF GPS coordinates in the format "[x, y, z]"
    parts = str(value).replace("[", "").replace("]", "").split(", ")
    degrees = float(parts[0])
    minutes_frac = Fraction(parts[1])
    seconds_frac = Fraction(parts[2])
    
    # Convert fractions to float
    minutes = minutes_frac.numerator / minutes_frac.denominator
    seconds = seconds_frac.numerator / seconds_frac.denominator
    
    return degrees, minutes, seconds

def convert_dms_to_dd(dms):
    # Function to convert degrees, minutes, seconds to decimal degrees
    degrees, minutes, seconds = dms
    dd = degrees + minutes / 60.0 + seconds / 3600.0
    return dd



def store_data_in_database(filename, boxes, geolocation):
    # Function to store the object detection results and geolocation in the SQLite database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Create the table if it doesn't exist
    cursor.execute(
        "CREATE TABLE IF NOT EXISTS object_detection_data ("
        "filename TEXT,"
        "x1 INTEGER, y1 INTEGER, x2 INTEGER, y2 INTEGER,"
        "object_type TEXT, probability REAL,"
        "latitude REAL, longitude REAL);"
    )

      # Generate a unique identifier using a combination of filename and UUID
    unique_id = f"{filename}_{uuid.uuid4()}"

    # Insert the data into the database
    for box in boxes:
        x1, y1, x2, y2, object_type, probability = box
        latitude, longitude = geolocation["latitude"], geolocation["longitude"]
        cursor.execute(
            "INSERT INTO object_detection_data (filename, x1, y1, x2, y2, object_type, probability, latitude, longitude) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (filename, x1, y1, x2, y2, object_type, probability, latitude, longitude),
        )

    conn.commit()
    conn.close()

# ... Rest of the code ...


def fetch_lat_lon_from_db_1(filename):
    try:
        connection = sqlite3.connect(db_path)
        cursor = connection.cursor()

        # Fetch one of the latitudes and longitudes for the given filename
        query = "SELECT latitude, longitude FROM object_detection_data WHERE filename = ? LIMIT 1"
        cursor.execute(query, (filename,))
        result = cursor.fetchone()

        return result

    except sqlite3.Error as error:
        print("Error fetching data from the database:", error)

    finally:
        if connection:
            connection.close()


def fetch_lat_lon_from_db():
    # Connect to the database
    conn = sqlite3.connect("object_detection_data.db")
    cursor = conn.cursor()

    # Fetch unique filenames and their count from the database
    cursor.execute("SELECT filename, COUNT(*) as count FROM object_detection_data GROUP BY filename")
    filenames_data = cursor.fetchall()

    # Fetch unique latitudes and longitudes from the database
    cursor.execute("SELECT DISTINCT latitude, longitude FROM object_detection_data")
    lat_lon_data = cursor.fetchall()

    # Close the database connection
    conn.close()

    return filenames_data, lat_lon_data

import sqlite3

# def fetch_img_names_and_counts_from_db():
#     # Connect to the database
#     db_path = "object_detection_data.db"  # Replace with the actual path to your SQLite database
#     conn = sqlite3.connect(db_path)
#     cursor = conn.cursor()

#     # Query to get image filenames and their counts from the database
#     query = """
#     SELECT filename, COUNT(*) AS count
#     FROM object_detection_data
#     GROUP BY filename
#     """

#     # Execute the query
#     cursor.execute(query)

#     # Fetch the results
#     results = cursor.fetchall()

#     # Separate the image filenames and counts into separate lists
#     img_names = [result[0] for result in results]
#     no_of_plastic = [result[1] for result in results]

#     # Close the database connection
#     cursor.close()
#     conn.close()

#     return img_names, no_of_plastic



def Bubble_map(db_name):
    # get the data from the sqlite database 
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    # Fetch unique filenames and their count with their unique lat and long from the database
    cursor.execute("SELECT filename, COUNT(filename), latitude, longitude FROM object_detection_data GROUP BY filename, latitude, longitude")
    rows = cursor.fetchall()

    # Create a dataframe from the rows
    df = pd.DataFrame(rows, columns=['filename', 'Plastic_count', 'latitude', 'longitude'])
    # Mapbox plot
    mapbox_fig = px.scatter_mapbox(df, lat='latitude', lon='longitude', size='Plastic_count',
                            color='Plastic_count', color_continuous_scale='plasma',
                            zoom=18, mapbox_style='open-street-map')
    mapbox_fig.update_traces(hovertemplate='<b>%{text}</b><br>' +
                                    'Plastic Count: %{marker.size:,}<br>' +
                                    'Latitude: %{lat}<br>' +
                                    'Longitude: %{lon}<br>',
                        text=df['filename'])
    
    # Bar plot
    bar_fig = px.bar(df, x='filename', y='Plastic_count', color='Plastic_count', color_continuous_scale='plasma')
    # add filename to the hover data
    bar_fig.update_traces(hovertemplate='<b>%{text}</b><br>' +
                                    'Plastic Count: %{y:,}<br>',
                        text=df['filename'])
    # line plot
    line_fig = px.line(df, x='filename', y='Plastic_count')
    
    # convert the plots to html
    mapbox_plot_div = mapbox_fig.to_html(full_html=False)
    bar_plot_div = bar_fig.to_html(full_html=False)
    line_plot_div = line_fig.to_html(full_html=False)


    return mapbox_plot_div, bar_plot_div, line_plot_div



"""

From here the Flask App starts
"""




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
    lat_lon = fetch_lat_lon_from_db_1(filename)
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

@app.route("/visualize")
def bubblemap():
    mapbox, bar, line = Bubble_map("object_detection_data.db")
    return render_template('visualize.html', mapbox_plot_div=mapbox, bar_plot_div=bar, line_plot_div=line)

# @app.route("/linechart")
# def linechart():
#     # Fetch data from the database (for example, using fetch_img_names_and_counts_from_db() function)
#     img_names, no_of_plastic = fetch_img_names_and_counts_from_db()

#     return jsonify(img_names, no_of_plastic)

# @app.route("/linecharts.html")
# def linecharts():
#     return render_template("linecharts.html")

@app.route("/locate")
def locate():
    return render_template("locate.html")



if __name__ == "__main__":
    main()
