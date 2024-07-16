import cv2
import numpy as np
import pandas as pd
import process

from datetime import datetime
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4


color_text = (128,128,0)

def tracejado(img, pos, orie, color, tickness = 1):
    try:
        altura, largura, _ = img.shape

        if orie == 'h':
            pt1 = (0, pos[1])
            pt2 = (largura, pos[1])

            x1, y1 = pt1

            dist = process.euclidean_dist(pt1, pt2)
            n = int(dist/10)

            for i in range(n):
                cv2.line(img, (x1, y1), (x1 + 5, y1), color, tickness)
                x1 += 15

        elif orie == 'v':
            pt1 = (pos[0], 0)
            pt2 = (pos[0], altura)

            x1, y1 = pt1

            dist = process.euclidean_dist(pt1, pt2)
            n = int(dist/10)

            for i in range(n):
                cv2.line(img, (x1, y1), (x1, y1 + 5), color, tickness)
                y1 += 15

        return img
    
    except:
        pass

def real_dist(A,B, coeficiente):
    # distancia real
    A = np.array(A)
    B = np.array(B)
    pix_dist = np.linalg.norm(A - B)

    return pix_dist*coeficiente

def draw_lines(frame, pts, med_hor, med_vert):
    try:
        hor_dist_pix = process.euclidean_dist(pts[0][0], pts[0][1])
        vert_dist_pix = process.euclidean_dist(pts[0][2], pts[0][3])

        coef_hor = med_hor/hor_dist_pix
        coef_vert = med_vert/vert_dist_pix


        # desenhando circulos nas pupilas
        cv2.circle(frame, pts[2][0], 3, (0, 255, 0), 1)
        cv2.circle(frame, pts[2][2], 3, (0, 255, 0), 1)

        DP = real_dist(pts[2][0], pts[2][2], coef_hor)

        PL_dist_left = real_dist(pts[2][1], pts[2][0], coef_vert)
        PL_dist_right = real_dist(pts[2][2], pts[2][3], coef_vert)
        ponte = real_dist(pts[0][5], pts[1][4], coef_hor)

        pupila_left_ponte = real_dist(pts[2][0], pts[2][4], coef_hor) 
        pupila_right_ponte = real_dist(pts[2][2], pts[2][4], coef_hor) 
        
        erro = abs(PL_dist_left - PL_dist_right)

        cv2.line(frame, pts[0][0], pts[0][1], (255,0,0), 1)
        cv2.line(frame, pts[0][2], pts[0][3], (0,255,0), 1)
        cv2.line(frame, pts[0][6], pts[0][7], (0,0,255), 1)
        cv2.line(frame, pts[1][0], pts[1][1], (100,100,0), 1)
        cv2.line(frame, pts[1][2], pts[1][3], (100,0,100), 1)    
        cv2.line(frame, pts[1][6], pts[1][7], (0,100,100), 1)
        cv2.line(frame, pts[0][4], pts[1][5], (50,50,50), 1)

        if (erro < 0.3*PL_dist_left) & (PL_dist_left < DP): 
            cv2.line(frame, pts[2][0], pts[2][1], color_text, 1)
            cv2.line(frame, pts[2][2], pts[2][3], color_text, 1)

        
        frame = tracejado(frame, pts[0][0], 'v', (255,0,0), 1)
        frame = tracejado(frame, pts[0][1], 'v', (255,0,0), 1)
        frame = tracejado(frame, pts[1][0], 'v', (100,100,0), 1)
        frame = tracejado(frame, pts[1][1], 'v', (100,100,0), 1)

        frame = tracejado(frame, pts[0][2], 'h', (0,255,0), 1)
        frame = tracejado(frame, pts[0][3], 'h', (0,255,0), 1)
        frame = tracejado(frame, pts[1][2], 'h', (100,0,100), 1)
        frame = tracejado(frame, pts[1][3], 'h', (100,0,100), 1)
        
        if(ponte < DP):
            cv2.line(frame, pts[0][5], pts[1][4], (0,255,0), 1)

        # df = pd.DataFrame(pts)
        # df.to_csv('static/pts_'+ cpf +'.csv', index=False)

        data = [['DP:', f"{(real_dist(pts[2][0], pts[2][2], coef_hor)):.3f}"],
                ['ALT esquerda:', f"{(real_dist(pts[2][0], pts[2][1], coef_vert)):.3f}"],
                ['ALT direita:', f"{(real_dist(pts[2][2], pts[2][3], coef_vert)):.3f}"],
                ['Ponte:', f"{ponte:.3f}"],
                ['DNP esquerda:', f"{pupila_right_ponte:.3f}"],
                ['DNP direita:', f"{pupila_left_ponte:.3f}"],
                ['Medida H. esquerda:', f"{(real_dist(pts[0][0], pts[0][1], coef_hor)):.3f}"],
                ['Medida H. direita:', f"{(real_dist(pts[1][0], pts[1][1], coef_hor)):.3f}"],
                ['Medida V. esquerda:', f"{(real_dist(pts[0][2], pts[0][3], coef_vert)):.3f}"],
                ['Medida V. direita:', f"{(real_dist(pts[1][2], pts[1][3], coef_vert)):.3f}"],
                ['MD esquerda:', f"{(real_dist(pts[0][6], pts[0][7], coef_hor)):.3f}"],
                ['MD direita:', f"{(real_dist(pts[1][6], pts[1][7], coef_hor)):.3f}"],
                ['Medida Externa:', f"{(real_dist(pts[0][4], pts[1][5], coef_hor)):.3f}"]]
        print(data)
        
        return frame, pts, data
    
    except Exception as e:
        print(f'{datetime.today()}, erro no drawlines: {e}')
        raise(e)

def create_pdf(data_dist, session):

    data_atual = datetime.now()
    date = data_atual.strftime("%d/%m/%Y")

    x = 325
    y = 750
    cnv = canvas.Canvas('static/dossie/pdf_'+ session['cpf'] +'.pdf', pagesize=A4)
    r, g, b = 77/255, 49/255, 78/255
    r2, g2, b2 = 97/255, 49/255, 84/255
    
    
    # cnv.drawImage('static/assets/pdf_background.png', 0, 0, width=600, height=1200)
    cnv.drawImage('static/assets/img/polvo.png', 90, 670, width=150, height=150, mask='auto')
    cnv.drawImage('static/assets/img1.png', 310, 660, width=260, height=150, mask='auto')
    cnv.drawImage('static/assets/img2.png', 30, 260, width=540, height=130, mask='auto')
    cnv.drawImage('static/assets/img3.png', 30, 100, width=540, height=130, mask='auto')
    cnv.drawImage('static/assets/img/polvo_mdl.png', 235, 5, width=127, height=100, mask='auto')
    cnv.drawImage('static/assets/img_first_'+ session['cpf'] +'.png', 35, 425, width=260, height=86)
    cnv.drawImage('static/assets/contour_img_'+ session['cpf'] +'.png', 310, 425, width=260, height=86)
    cnv.drawImage('static/assets/'+ session['cpf'] +'.png', 35, 520, width=535, height=120)


    cnv.setFont("Helvetica-Bold", 20)
    cnv.setFillColorRGB(r,g,b)
    cnv.drawString(325, 780,'DADOS DO CLIENTE')
    cnv.drawString(230, 200, 'I.R. INDICADO')

    cnv.setFillColorRGB(r2,g2,b2)
    cnv.setFont("Helvetica-Bold", 11)
    cnv.drawString(x, y, f"Nome:" )
    cnv.drawString(x, y-20, f"CPF:" )
    cnv.drawString(x, y-40, f"Telefone:" )
    cnv.drawString(x, y-60, f"Data:" )
    cnv.drawString(x, y-80, f"TSO:" )

    cnv.setFillColorRGB(0,0,0)
    cnv.drawString(x+90, y, f"{session['nome']}" )
    cnv.drawString(x+90, y-20, f"{session['cpf']}" )
    cnv.drawString(x+90, y-40, f"{session['telefone']}")
    cnv.drawString(x+90, y-60, f"{date}")
    cnv.drawString(x+90, y-80, f"{session['tso']}")

    cnv.setFillColorRGB(r2,g2,b2)

    x = 60
    y = 160

    cnv.drawString(x, y, f"OD-Esf:" )
    cnv.drawString(x, y-24, f"OE-Esf:" )
    cnv.drawString(x, y-48, f"AD:" )

    cnv.drawString(x+130, y, f"OD-Cil:" )
    cnv.drawString(x+130, y-24, f"OE-Cil:" )
    cnv.drawString(x+130, y-48, f"PL:" )

    cnv.drawString(x+260, y, f"OD-Eixo:" )
    cnv.drawString(x+260, y-24, f"OE-Eixo:" )
    cnv.drawString(x+260, y-48, f"Prisma:" )


    cnv.drawString(x+390, y, f"OD-DNP:" )
    cnv.drawString(x+390, y-24, f"OE-DNP:" )
    
    
    
    cnv.setFont("Helvetica-Bold", 11)

    x += 60
    
    cnv.drawString(x, y, f"{session['od_esf']}" )
    cnv.drawString(x+130, y, f"{session['od_cil']}" )
    cnv.drawString(x+260, y, f"{session['od_eixo']}" )
    cnv.drawString(x+390, y, f"{session['od_dnp']}" )

    y -= 24

    cnv.drawString(x, y, f"{session['oe_esf']}" )
    cnv.drawString(x+130, y, f"{session['oe_cil']}" )
    cnv.drawString(x+260, y, f"{session['oe_eixo']}" )
    cnv.drawString(x+390, y, f"{session['oe_dnp']}" )

    y -= 24

    cnv.drawString(x, y, f"{session['ad']}" )
    cnv.drawString(x+130, y, f"{session['pl']}" )
    cnv.drawString(x+260, y, f"{session['prisma']}" )

    y = 370
    x = 50
    aux = 0

    cores = [(70/255,130/255,180/255), (0,0,128/255), (0,128/255,128/255), (0,128/255,0), (128/255,128/255,0), (128/255,0,0), (255/255,0,0), (255/255,165/255,0), (80/255,220/255,0), (0,255/255,0), (0,255/255,255/255), (0,0,255/255), (255/255,0,255/255)]


    # Larguras das colunas
    col_widths = [110, 150]

    esquerda = 1,4,6,8,10
    direita = 2,5,7,9,11
    centro = 0,3,12

    #Desenha a tabela
    for j in centro:
        r,g,b = cores[aux]
        cnv.setFillColorRGB(r,g,b)
        cnv.drawString(x, y, data_dist[j][0])
        cnv.drawString(x+col_widths[0], y, data_dist[j][1])
        aux += 1
        y -= 24

    y = 370

    for j in esquerda:
        r,g,b = cores[aux]
        cnv.setFillColorRGB(r,g,b)
        cnv.drawString(x+170, y, data_dist[j][0])
        cnv.drawString(x+col_widths[0]+170, y, data_dist[j][1])
        aux += 1
        y -= 24

    y = 370

    for j in direita:
        r,g,b = cores[aux]
        cnv.setFillColorRGB(r,g,b)
        cnv.drawString(x+340, y, data_dist[j][0])
        cnv.drawString(x+col_widths[0]+340, y, data_dist[j][1])
        aux += 1
        y -= 24


    cnv.save()


        
       

