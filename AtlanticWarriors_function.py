
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
db_path = "object_detection_data.db"
