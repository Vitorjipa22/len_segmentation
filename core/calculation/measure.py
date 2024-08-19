import cv2

import numpy as np
from datetime import datetime
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4

from core.lib.utils import euclidean_dist
from core.lib.logger import logger


class Measure:
    def __init__(self) -> None:
        """
        Initialize the Measure class with default settings.
        """
        self.color_text = (128, 128, 0)

    def predict(self, frame: np.ndarray, pts: list, med_hor: float, med_vert: float) -> tuple:
        """
        Analyze the input frame and points to predict various measurements.

        Args:
            frame (np.ndarray): Image frame to process.
            pts (list): List of points for distance calculations.
            med_hor (float): Horizontal measurement calibration.
            med_vert (float): Vertical measurement calibration.

        Returns:
            tuple: The processed frame, the points, and the calculated data.
        """
        try:
            # Calculate pixel distances
            hor_dist_pix = euclidean_dist(pts[0][0], pts[0][1])
            vert_dist_pix = euclidean_dist(pts[0][2], pts[0][3])

            # Calculate conversion coefficients from pixels to real units
            coef_hor = med_hor / hor_dist_pix
            coef_vert = med_vert / vert_dist_pix

            # Calculate real distances
            DP = self.real_dist(pts[2][0], pts[2][2], coef_hor)
            PL_dist_left = self.real_dist(pts[2][1], pts[2][0], coef_vert)
            PL_dist_right = self.real_dist(pts[2][2], pts[2][3], coef_vert)
            ponte = self.real_dist(pts[0][5], pts[1][4], coef_hor)
            pupila_left_ponte = self.real_dist(pts[2][0], pts[2][4], coef_hor)
            pupila_right_ponte = self.real_dist(pts[2][2], pts[2][4], coef_hor)

            # Draw various lines on the frame
            self.draw_elements(frame, pts, PL_dist_left, PL_dist_right, DP, ponte)

            # Draw dashed lines on the frame
            self.draw_dashed_lines(frame, pts)

            # Compile measurement data
            data = [
                ['DP:', f"{DP:.3f}"],
                ['ALT esquerda:', f"{PL_dist_left:.3f}"],
                ['ALT direita:', f"{PL_dist_right:.3f}"],
                ['Ponte:', f"{ponte:.3f}"],
                ['DNP esquerda:', f"{pupila_right_ponte:.3f}"],
                ['DNP direita:', f"{pupila_left_ponte:.3f}"],
                ['Medida H. esquerda:', f"{hor_dist_pix * coef_hor:.3f}"],
                ['Medida H. direita:', f"{euclidean_dist(pts[1][0], pts[1][1]) * coef_hor:.3f}"],
                ['Medida V. esquerda:', f"{euclidean_dist(pts[0][2], pts[0][3]) * coef_vert:.3f}"],
                ['Medida V. direita:', f"{euclidean_dist(pts[1][2], pts[1][3]) * coef_vert:.3f}"],
                ['MD esquerda:', f"{euclidean_dist(pts[0][6], pts[0][7]) * coef_hor:.3f}"],
                ['MD direita:', f"{euclidean_dist(pts[1][6], pts[1][7]) * coef_hor:.3f}"],
                ['Medida Externa:', f"{euclidean_dist(pts[0][4], pts[1][5]) * coef_hor:.3f}"],
            ]

            print(data)

            return frame, pts, data

        except Exception as e:
            logger.error(f"Error in measure predict method: {e}")
            raise e

    def dashed_lines(self, img: np.ndarray, pos: tuple, orientation: str, color: tuple, thickness: int = 1) -> np.ndarray:
        """
        Draw dashed lines on the image either horizontally or vertically.

        Args:
            img (np.ndarray): Image to draw on.
            pos (tuple): Position of the dashed line.
            orientation (str): Orientation of the line ('h' for horizontal, 'v' for vertical).
            color (tuple): Color of the line.
            thickness (int): Thickness of the line.

        Returns:
            np.ndarray: Image with dashed lines.
        """
        try:
            height, width, _ = img.shape

            if orientation == 'h':
                pt1 = (0, pos[1])
                pt2 = (width, pos[1])
                x1, y1 = pt1

                # Draw horizontal dashed line
                dist = euclidean_dist(pt1, pt2)
                n = int(dist / 10)

                for i in range(n):
                    cv2.line(img, (x1, y1), (x1 + 5, y1), color, thickness)
                    x1 += 15

            elif orientation == 'v':
                pt1 = (pos[0], 0)
                pt2 = (pos[0], height)
                x1, y1 = pt1

                # Draw vertical dashed line
                dist = euclidean_dist(pt1, pt2)
                n = int(dist / 10)

                for i in range(n):
                    cv2.line(img, (x1, y1), (x1, y1 + 5), color, thickness)
                    y1 += 15

            return img

        except Exception as e:
            logger.error(f"Error in dashed_lines method: {e}")
            raise e

    def real_dist(self, A: tuple, B: tuple, coeficiente: float) -> float:
        """
        Calculate the real-world distance between two points, given a scaling coefficient.

        Args:
            A (tuple): First point (x, y).
            B (tuple): Second point (x, y).
            coeficiente (float): Scaling coefficient to convert pixel distance to real-world distance.

        Returns:
            float: Real-world distance.
        """
        A = np.array(A)
        B = np.array(B)
        pix_dist = np.linalg.norm(A - B)
        return pix_dist * coeficiente

    def create_pdf(self, data_dist: list, session: dict) -> None:
        """
        Create a PDF report based on the provided data and session information.

        Args:
            data_dist (list): List of distance data to include in the report.
            session (dict): Dictionary containing session information (e.g., client data).
        """
        try:
            data_atual = datetime.now()
            date = data_atual.strftime("%d/%m/%Y")

            # Setup PDF canvas
            cnv = canvas.Canvas('static/dossie/pdf_' + session['cpf'] + '.pdf', pagesize=A4)
            r, g, b = 77 / 255, 49 / 255, 78 / 255
            r2, g2, b2 = 97 / 255, 49 / 255, 84 / 255

            # Draw images on the PDF
            cnv.drawImage('static/assets/img/polvo.png', 90, 670, width=150, height=150, mask='auto')
            # Additional drawing of images goes here...

            # Draw client data on the PDF
            cnv.setFont("Helvetica-Bold", 20)
            cnv.setFillColorRGB(r, g, b)
            cnv.drawString(325, 780, 'DADOS DO CLIENTE')

            # Populate client-specific fields
            cnv.setFont("Helvetica-Bold", 11)
            cnv.setFillColorRGB(0, 0, 0)
            cnv.drawString(415, 750, session['nome'])
            cnv.drawString(415, 730, session['cpf'])
            cnv.drawString(415, 710, session['telefone'])
            cnv.drawString(415, 690, date)
            cnv.drawString(415, 670, session['tso'])

            # Draw distance data table
            self.draw_data_table(cnv, data_dist)

            cnv.save()

        except Exception as e:
            logger.error(f"Error in create_pdf method: {e}")
            raise e

    def draw_elements(self, frame: np.ndarray, pts: list, PL_dist_left: float, PL_dist_right: float, DP: float, Ponte: float) -> None:
        """
        Draw various elements on the frame.

        Args:
            frame (np.ndarray): Image frame to draw elements on.
            pts (list): List of points to define the lines.
        """
        # Calculate error between left and right distances
        erro = abs(PL_dist_left - PL_dist_right)

        cv2.line(frame, pts[0][0], pts[0][1], (255,0,0), 1)
        cv2.line(frame, pts[0][2], pts[0][3], (0,255,0), 1)
        cv2.line(frame, pts[0][6], pts[0][7], (0,0,255), 1)
        cv2.line(frame, pts[1][0], pts[1][1], (100,100,0), 1)
        cv2.line(frame, pts[1][2], pts[1][3], (100,0,100), 1)    
        cv2.line(frame, pts[1][6], pts[1][7], (0,100,100), 1)
        cv2.line(frame, pts[0][4], pts[1][5], (50,50,50), 1)

        if (erro < 0.3*PL_dist_left) & (PL_dist_left < DP): 
            cv2.line(frame, pts[2][0], pts[2][1], self.color_text, 1)
            cv2.line(frame, pts[2][2], pts[2][3], self.color_text, 1)

        if(Ponte < DP):
            cv2.line(frame, pts[0][5], pts[1][4], (0,255,0), 1)
        
        # Draw circles on pupil points
            cv2.circle(frame, pts[2][0], 3, (0, 255, 0), 1)
            cv2.circle(frame, pts[2][2], 3, (0, 255, 0), 1)

    def draw_dashed_lines(self, frame: np.ndarray, pts: list) -> None:
        """
        Draw dashed lines on the frame.

        Args:
            frame (np.ndarray): Image frame to draw dashed lines on.
            pts (list): List of points to define the dashed lines.
        """
        # Drawing dashed line logic goes here...frame = self.tracejado(frame, pts[0][0], 'v', (255,0,0), 1)
        frame = self.dashed_lines(frame, pts[0][1], 'v', (255,0,0), 1)
        frame = self.dashed_lines(frame, pts[1][0], 'v', (100,100,0), 1)
        frame = self.dashed_lines(frame, pts[1][1], 'v', (100,100,0), 1)

        frame = self.dashed_lines(frame, pts[0][2], 'h', (0,255,0), 1)
        frame = self.dashed_lines(frame, pts[0][3], 'h', (0,255,0), 1)
        frame = self.dashed_lines(frame, pts[1][2], 'h', (100,0,100), 1)
        frame = self.dashed_lines(frame, pts[1][3], 'h', (100,0,100), 1)

    def draw_data_table(self, cnv: canvas.Canvas, data_dist: list) -> None:
        """
        Draw the distance data table on the PDF.

        Args:
            cnv (canvas.Canvas): PDF canvas object.
            data_dist (list): List of distance data to populate the table.
        """
        # Table drawing logic goes here...
