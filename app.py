import io
import cv2 as cv 
import numpy as np
import pandas as pd
import sys
import os
from flask import Flask, render_template, request, session, send_file

from core.calculation.measure import Measure
from core.image_analysis.process import Process
from core.image_analysis.detect import LenDetector
from core.image_analysis.segment import Segmentator

app = Flask(__name__)
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
app.secret_key = 'super secret'

# Adds the root path to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

@app.route('/generate_pdf', methods=['POST'])
def generate_pdf():
    """
    Generates a PDF from a processed image and saves the generated file in a specific folder.
    
    This endpoint loads the image and contour data from the user's session, processes the information,
    and generates a report in PDF format.
    """
    # Load images and contour data
    frame = cv.imread('static/assets/img_cropped_' + session['cpf'] + '.png')
    contour_img = cv.imread('static/assets/contour_img_' + session['cpf'] + '.png')
    contour_img = np.array(cv.cvtColor(contour_img, cv.COLOR_BGR2GRAY))
    pts = Measure.read_csv_pts('static/pts_' + session['cpf'] + '.csv', [1, 2, 3])

    # Load and process the mesh
    mesh = pd.read_csv('static/mesh_' + session['cpf'] + '.csv')
    mesh = np.array(mesh)
    
    # Draw lines and process the image
    img, _, session['data'] = Measure.draw_lines(frame.copy(), contour_img, mesh, pts, session['cpf'])
    
    # Create the PDF
    Measure.create_pdf(session['data'], name=session['name'], CPF=session['cpf'], tel=session['cpf'], path='static/dossier/pdf_' + session['cpf'] + '.pdf')

    return render_template('page9.html', cpf=session['cpf'])

@app.route('/save_frame_head', methods=['POST', 'GET'])
def save_frame_head(): 
    """
    Processes a received image, detects, segments, and performs measurements.

    This endpoint receives an image via POST, performs detection and segmentation on the image,
    and calculates measurements based on provided parameters.
    """
    if 'image' not in request.files or request.files['image'].filename == '':
        return "No image part or no selected file", 400
    
    file = request.files['image']

    try:
        med_hor = float(request.form.get('horizontal_measure'))
        med_vert = float(request.form.get('vertical_measure'))
    except (TypeError, ValueError):
        return "Invalid float values", 400
    
    # Initialize detection, segmentation, and processing classes
    len_detector = LenDetector()
    segmentator = Segmentator()
    process = Process()
    measure = Measure()

    # Read the received image and decode
    image_stream = file.read()
    np_img = np.frombuffer(image_stream, np.uint8)
    image = cv.imdecode(np_img, cv.IMREAD_COLOR)

    # Copy the original image for processing
    frame_ori = image.copy()

    # Detect bounding boxes
    boxes = len_detector.predict(frame_ori.copy())

    # Perform segmentation based on the detected boxes
    mask = segmentator.predict(frame_ori.copy(), boxes)

    # Predict points based on the segmented mask
    pts = process.predict(frame_ori, mask)

    # Perform measurements and save the result
    result_img, pts, data = measure.predict(frame_ori, pts, med_hor, med_vert)
    image = np.asarray(result_img)

    # Encode the image to JPEG format and return the file
    _, img_encoded = cv.imencode('.jpg', image)
    img_io = io.BytesIO(img_encoded)
    
    return send_file(img_io, mimetype='image/jpeg')

@app.route('/download_pdf', methods=['POST'])
def download_pdf():
    """
    Generates and downloads a PDF based on the information stored in the user's session.

    This endpoint reads processed images, points, and contour data to generate a PDF,
    which is then sent as a download to the user.
    """
    data_index = []

    # Load the image and contour
    frame = cv.imread('static/assets/frame_' + session['cpf'] + '.png')
    contour_img = cv.imread('static/assets/contour_img_' + session['cpf'] + '.png')
    contour_img = np.array(cv.cvtColor(contour_img, cv.COLOR_BGR2GRAY))
    pts = Measure.read_csv_pts('static/pts_' + session['cpf'] + '.csv', [1, 2, 3])

    # Load and process the mesh
    mesh = pd.read_csv('static/mesh_' + session['cpf'] + '.csv')
    mesh = np.array(mesh)
    
    # Draw lines and process the image
    img, _, session['data'] = Measure.draw_lines(frame.copy(), contour_img, mesh, pts, session['cpf'])

    # Populate the data index for the report
    data_index.extend([session['od_sph'], session['od_cyl'], session['od_axis'], session['od_dnp'],
                       session['oe_sph'], session['oe_cyl'], session['oe_axis'], session['oe_dnp'],
                       session['add'], session['pl'], session['prism']])

    # Create the PDF and send it for download
    Measure.create_pdf(session['data'], session)
    
    return send_file('static/dossier/pdf_' + session['cpf'] + '.pdf', as_attachment=True)


if __name__ == "__main__":
    app.run(debug=True)
