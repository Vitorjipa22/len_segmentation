import io
import measure
import cv2 as cv 
import numpy as np
import pandas as pd
import measure
import process
import matplotlib.pyplot as plt

from datetime import datetime
from process import Len_calculator
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from flask import Flask, render_template, request, session, send_file

app = Flask(__name__)
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
app.secret_key = 'super secret'

@app.route('/gerar_pdf', methods=['POST'])
def gerar_pdf():
    frame = cv.imread('static/assets/img_recortada_'+ session['cpf'] +'.png')
    contour_img = cv.imread('static/assets/contour_img_'+ session['cpf'] +'.png')
    contour_img = np.array(cv.cvtColor(contour_img, cv.COLOR_BGR2GRAY))
    pts = measure.read_csv_pts('static/pts_'+ session['cpf'] +'.csv', [1,2,3])

    mesh = pd.read_csv('static/mesh_'+ session['cpf'] +'.csv')
    mesh = np.array(mesh)
    img, _, session['data'] = measure.draw_lines(frame.copy(), contour_img, mesh, pts, session['cpf'])

    cv.imwrite('static/assets/'+ session['cpf'] +'.png', img)
    measure.create_pdf(session['data'], nome=session['nome'], CPF=session['cpf'], tel=session['cpf'], path= 'static/dossie/pdf_'+ session['cpf'] +'.pdf')
    # iris2.create_pdf([], nome=dados[0], CPF=dados[1], tel=dados[2], path= 'static/dossie/pdf_'+ dados[1] +'.pdf')
    # iris2.create_pdf(path= 'static/dossie/pdf_'+ '03050878274' +'.pdf')

    return render_template('page9.html', cpf = session['cpf'])

@app.route('/save_frame_head', methods=['POST', 'GET'])
def save_frame_head(): 
    len = Len_calculator(segment_type = 'vit_b', checkpoint_path='models/sam_vit_b_01ec64.pth')

    file = request.files['image']

    if 'image' not in request.files:
        return "No image part", 400
    
    if file.filename == '':
        return "No selected file", 400
    
    if file:
        # Leia a imagem
        image_stream = file.read()
        np_img = np.frombuffer(image_stream, np.uint8)
        image = cv.imdecode(np_img, cv.IMREAD_COLOR)

        # Leia os valores float
        try:
            med_hor = float(request.form.get('medida_horizontal'))
            med_vert = float(request.form.get('medida_vertical'))

        except (TypeError, ValueError):
            return "Invalid float values", 400
        
        frame_ori = image.copy()
        mask = len.predict(frame_ori.copy())

        try:
            mesh_points = process.getting_mesh_points(frame_ori.copy())
  
            img_bin, centroids, largest_area_idxs, labels = process.pre_process(mask)
            contour_img, sides, centr, pts = process.lens_localization(img_bin, largest_area_idxs, labels, centroids)
            
            pts = process.find_biggest_distance(sides, centr, pts)

        except Exception as e:
            print('Erro na segmentação da lente: ', e)

        pts = process.adding_additional_points(pts, mesh_points, contour_img)

        result_img, pts, data = measure.draw_lines(frame_ori, pts, med_hor, med_vert)
        image = result_img


        _, img_encoded = cv.imencode('.jpg', image)
        img_io = io.BytesIO(img_encoded)
        
        return send_file(img_io, mimetype='image/jpeg')

@app.route('/baixar_pdf', methods=['POST'])
def baixar_pdf():
    data_index = list()

    # session['nome'] = 'teste'
    # session['cpf'] = '888.888.888-88'
    # session['telefone'] = 'teste'
    # session['tso'] = 'teste'
    # session['od_esf'] = 'teste'
    # session['od_cil'] = 'teste'
    # session['od_eixo'] = 'teste'
    # session['od_dnp'] = 'teste'
    # session['oe_esf'] = 'teste'
    # session['oe_cil'] = 'teste'
    # session['oe_eixo'] = 'teste'
    # session['oe_dnp'] = 'teste'
    # session['ad'] = 'teste'
    # session['pl'] = 'teste'
    # session['prisma'] = 'teste'

    frame = cv.imread('static/assets/frame_'+ session['cpf'] +'.png')
    contour_img = cv.imread('static/assets/contour_img_'+ session['cpf'] +'.png')
    contour_img = np.array(cv.cvtColor(contour_img, cv.COLOR_BGR2GRAY))
    pts = measure.read_csv_pts('static/pts_'+ session['cpf'] +'.csv', [1,2,3])

    mesh = pd.read_csv('static/mesh_'+ session['cpf'] +'.csv')
    mesh = np.array(mesh)
    img, _, session['data'] = measure.draw_lines(frame.copy(), contour_img, mesh, pts, session['cpf'])

    cv.imwrite('static/assets/'+ session['cpf'] +'.png', img)

    # retirar para subir

    
    
    data_index.append(session['od_esf'])
    data_index.append(session['od_cil'])
    data_index.append(session['od_eixo'])
    data_index.append(session['od_dnp'])
    data_index.append(session['oe_esf'])
    data_index.append(session['oe_cil'])
    data_index.append(session['oe_eixo'])
    data_index.append(session['oe_dnp'])
    data_index.append(session['ad'])
    data_index.append(session['pl'])
    data_index.append(session['prisma'])

    measure.create_pdf(session['data'], session)

    return send_file('static/dossie/pdf_'+ session['cpf'] +'.pdf', as_attachment=True)


if __name__ == "__main__":
    app.run(debug=True)
