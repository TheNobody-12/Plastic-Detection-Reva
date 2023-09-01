import onnxruntime as ort
from flask import request, Flask, jsonify, render_template,session,redirect
# import session
from flask_sqlalchemy import SQLAlchemy
import bcrypt
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
geolocator = Nominatim(user_agent="object-detection-app_1}")

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



# def store_data_in_database(filename, boxes, geolocation):
#     # Function to store the object detection results and geolocation in the SQLite database
#     conn = sqlite3.connect(app.config['SQLALCHEMY_DATABASE_URI'])
#     cursor = conn.cursor()

#     # Create the table if it doesn't exist
#     cursor.execute(
#         """ 
#         CREATE TABLE users (
#             id SERIAL PRIMARY KEY,
#             name VARCHAR(100) NOT NULL,
#             email VARCHAR(100) UNIQUE NOT NULL,
#             password VARCHAR(100) NOT NULL
#         );

#         CREATE TABLE object_detection_data (
#             id SERIAL PRIMARY KEY,
#             user_id INT REFERENCES users(id),
#             filename TEXT,
#             x1 INTEGER,
#             y1 INTEGER,
#             x2 INTEGER,
#             y2 INTEGER,
#             object_type TEXT,
#             probability REAL,
#             latitude REAL,
#             longitude REAL
#         );
# """    )

#       # Generate a unique identifier using a combination of filename and UUID
#     unique_id = f"{filename}_{uuid.uuid4()}"

#     # Insert the data into the database
#     for box in boxes:
#         x1, y1, x2, y2, object_type, probability = box
#         latitude, longitude = geolocation["latitude"], geolocation["longitude"]
#         cursor.execute(
#             "INSERT INTO object_detection_data (filename, x1, y1, x2, y2, object_type, probability, latitude, longitude) "
#             "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
#             (filename, x1, y1, x2, y2, object_type, probability, latitude, longitude),
#         )

#     conn.commit()
#     conn.close()

# ... Rest of the code ...




def fetch_lat_lon_from_db_1(filename):
    try:
        connection = sqlite3.connect(app.config['SQLALCHEMY_DATABASE_URI'])
        cursor = connection.cursor()

        # Fetch one of the latitudes and longitudes for the given filename
        query = "SELECT latitude, longitude FROM reva_object_data WHERE filename = ? LIMIT 1"
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
    conn = sqlite3.connect("reva_object_data.db")
    cursor = conn.cursor()

    # Fetch unique filenames and their count from the database
    cursor.execute("SELECT filename, COUNT(*) as count FROM reva_object_data GROUP BY filename")
    filenames_data = cursor.fetchall()

    # Fetch unique latitudes and longitudes from the database
    cursor.execute("SELECT DISTINCT latitude, longitude FROM reva_object_data")
    lat_lon_data = cursor.fetchall()

    # Close the database connection
    conn.close()

    return filenames_data, lat_lon_data



def Bubble_map(db_name):
    # get the data from the sqlite database 
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    # Fetch unique filenames and their count with their unique lat and long from the database
    cursor.execute("SELECT filename, COUNT(filename), latitude, longitude FROM reva_object_data GROUP BY filename, latitude, longitude")
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
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///REVA.db'
db_s = SQLAlchemy(app)
app.secret_key = 'secret_key'

class User(db_s.Model):
    id = db_s.Column(db_s.Integer, primary_key=True)
    name = db_s.Column(db_s.String(100), nullable=False)
    email = db_s.Column(db_s.String(100), unique=True)
    password = db_s.Column(db_s.String(100))
    reva_object_data = db_s.relationship('ObjectDetectionData', backref='user', lazy=True)

    def __init__(self, email, password, name):
        self.name = name
        self.email = email
        self.password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

    def check_password(self, password):
        return bcrypt.checkpw(password.encode('utf-8'), self.password.encode('utf-8'))

class ObjectDetectionData(db_s.Model):
    id = db_s.Column(db_s.Integer, primary_key=True)
    user_id = db_s.Column(db_s.Integer, db_s.ForeignKey('user.id'), nullable=False)
    filename = db_s.Column(db_s.Text)
    x1 = db_s.Column(db_s.Integer)
    y1 = db_s.Column(db_s.Integer)
    x2 = db_s.Column(db_s.Integer)
    y2 = db_s.Column(db_s.Integer)
    object_type = db_s.Column(db_s.Text)
    probability = db_s.Column(db_s.Float)
    latitude = db_s.Column(db_s.Float)
    longitude = db_s.Column(db_s.Float)

    def __init__(self, user_id, filename, x1, y1, x2, y2, object_type, probability, latitude, longitude):
        self.user_id = user_id
        self.filename = filename
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.object_type = object_type
        self.probability = probability
        self.latitude = latitude
        self.longitude = longitude



with app.app_context():
    db_s.create_all()


@app.route('/register',methods=['GET','POST'])
def register():
    if request.method == 'POST':
        # handle request
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']

        new_user = User(name=name,email=email,password=password)
        # add the new user to the database
        db_s.session.add(new_user)    
        db_s.session.commit()
        return redirect('/login')

    return render_template('register.html')

@app.route('/login',methods=['GET','POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        user = User.query.filter_by(email=email).first()
        
        if user and user.check_password(password):
            session['email'] = user.email
            return redirect('/dashboard')
        else:
            return render_template('login.html',error='Invalid user')

    return render_template('login.html')


@app.route('/dashboard')
def dashboard():
    if session['email']:
        user = User.query.filter_by(email=session['email']).first()
        return render_template('index.html',user=user)
    
    return redirect('/login')

@app.route('/logout')
def logout():
    session.pop('email',None)
    return redirect('/login')


def main():
    # serve(app, host='0.0.0.0', port=8080)
    app.run(debug=True,port=8080)


@app.route("/")
def root():
    """
    Site main page handler function.
    :return: Content of index.html file
    """

    # with open("templates/index.html") as file:
    #     return file.read()
    return render_template("home.html")

@app.route("/home")
def home():
    return render_template("home.html")

@app.route("/predict")
def predict():
    return render_template("predict.html")

@app.route("/login_reg")
def login_reg():
    return render_template("login.html")


def get_user_id_from_session():
    # Check if the 'user_id' key exists in the session
    if 'user_id' in session:
        # Return the user ID stored in the session
        return session['user_id']
    else:
        # Return None if the user is not authenticated
        return None
def store_data_in_database(user_id, filename, boxes, geolocation):
    # Function to store the object detection results and geolocation in the SQLite database
    conn = sqlite3.connect(app.config['SQLALCHEMY_DATABASE_URI'])
    cursor = conn.cursor()

    # Generate a unique identifier using a combination of filename and UUID
    unique_id = f"{filename}_{uuid.uuid4()}"

    # Insert the data into the database
    for box in boxes:
        x1, y1, x2, y2, object_type, probability = box
        latitude, longitude = geolocation["latitude"], geolocation["longitude"]
        cursor.execute(
            "INSERT INTO reva_object_data (user_id, filename, x1, y1, x2, y2, object_type, probability, latitude, longitude) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (user_id, filename, x1, y1, x2, y2, object_type, probability, latitude, longitude),
        )

    conn.commit()
    conn.close()


@app.route("/detect", methods=["POST"])
def detect():
    # Get the user ID associated with the session (you may need to implement user authentication and session handling)
    user_id = get_user_id_from_session()  # Implement this function

    if user_id is None:
        return jsonify({"error": "User not authenticated"})

    buf = request.files["image_file"]
    filename = buf.filename
    print(filename)
    boxes = detect_objects_on_image(buf.stream)

    # Get geolocation from the image metadata
    geolocation = get_image_geolocation(buf)

    # Store the data in the SQLite database
    store_data_in_database(user_id, filename, boxes, geolocation)

    return jsonify(boxes)

# @app.route("/detect", methods=["POST"])
# def detect():
#     """
#     Handler of /detect POST endpoint
#     Receives uploaded file with a name "image_file", passes it
#     through YOLOv8 object detection network and returns an array
#     of bounding boxes along with geolocation from the image metadata.
#     The data is also stored in a SQLite database.
#     :return: a JSON array of objects containing bounding boxes in format [[x1, y1, x2, y2, object_type, probability], ...]
#     """
#     buf = request.files["image_file"]
#     filename = buf.filename
#     print(filename)
#     boxes = detect_objects_on_image(buf.stream)

#     # Get geolocation from the image metadata
#     geolocation = get_image_geolocation(buf)

#     # Store the data in the SQLite database
#     for box in boxes:
#             x1, y1, x2, y2, object_type, probability = box
#             latitude, longitude = geolocation["latitude"], geolocation["longitude"]
#             new_data = ObjectDetectionData(filename=filename, x1=x1, y1=y1, x2=x2, y2=y2, object_type=object_type, probability=probability, latitude=latitude, longitude=longitude)
#     # return jsonify(boxes, geolocation)
#     return jsonify(boxes)

# create different route for the database
@app.route("/database", methods=["GET"])
def database():
    """
    Handler of /database GET endpoint
    Retrieves data from the SQLite database and returns it as a JSON array
    :return: a JSON array of objects containing bounding boxes in format [[x1, y1, x2, y2, object_type, probability], ...]
    """
    conn = sqlite3.connect(app.config['SQLALCHEMY_DATABASE_URI'])
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM reva_object_data")
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

    d_b = {
        "filenames": filenames_data,
        "lat_lon": lat_lon_data
    }

    return jsonify(d_b)

@app.route("/db")
def db():
    return render_template("db.html")

@app.route("/visualize")
def bubblemap():
    mapbox, bar, line = Bubble_map("reva_object_data.db")
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