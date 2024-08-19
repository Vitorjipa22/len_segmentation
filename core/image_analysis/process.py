import cv2
import numpy as np
import mediapipe as mp
from core.lib.logger import logger
from core.lib.utils import euclidean_dist

class Process:
    def __init__(self):
        self.RIGHT_IRIS = [474, 475, 476, 477]
        self.LEFT_IRIS = [469, 470, 471, 472]

    def predict(self, frame, mask):
        try:
            mesh_points = self.getting_mesh_points(frame)
  
            img_bin, centroids, largest_area_idxs, labels = self.pre_process(mask)
            contour_img, sides, centr, pts = self.lens_localization(img_bin, largest_area_idxs, labels, centroids)
            
            pts = self.find_biggest_distance(sides, centr, pts)
            pts = self.adding_additional_points(pts, mesh_points, contour_img)

            return pts

        except Exception as e:
            logger.error('Erro na segmentação da lente: ', e)

    def getting_mesh_points(self, frame):
        mp_face_mesh = mp.solutions.face_mesh

        with mp_face_mesh.FaceMesh(
            max_num_faces = 1,
            refine_landmarks = True,
            min_detection_confidence = 0.5,
            min_tracking_confidence = 0.5
        ) as face_mesh:

            results = face_mesh.process(frame)

            if results.multi_face_landmarks:
                img_h, img_w = frame.shape[:2]

                mesh_points = np.array([np.multiply([p.x, p.y], [img_w, img_h]).astype(int) for p in results.multi_face_landmarks[0].landmark])

            else:
                logger.error('No face detected.')

        return mesh_points

    def pre_process(self, img):
        """
        Process the largest lens areas in a binary image.

        Parameters:
        img (numpy.ndarray): Input binary image where lens regions need to be segmented.

        Returns:
        tuple: A tuple containing:
            - img (numpy.ndarray): Original input image.
            - img_bin (numpy.ndarray): Binary version of the input image.
            - centroids (numpy.ndarray): Array of centroids for each labeled region.
            - largest_area_idxs (numpy.ndarray): Indices of the largest segmented areas.
            - labels (numpy.ndarray): Label matrix where each region in the image is labeled with a unique integer.
        """
        # Convert the image to uint8 type
        img_bin = img.astype(np.uint8)
        img_len = img_bin.copy()

        # Perform connected component analysis
        _, labels, stats, centroids = cv2.connectedComponentsWithStats(img_len)
        
        # Get the areas of the components and find the indices of the two largest areas
        areas = stats[1:, cv2.CC_STAT_AREA]
        largest_area_idxs = np.argsort(areas)[-2:] + 1
        
        # Create a binary image containing only the largest areas
        img_len = np.zeros_like(img_len)
        for idx in largest_area_idxs:
            img_len[labels == idx] = 255

        # Find contours of the largest areas
        contours, _ = cv2.findContours(img_len, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour_img = np.zeros_like(img_len)
        aux = np.zeros_like(img_len)
        cv2.drawContours(aux, contours, -1, 255, 1)

        # Compute convex hulls for each contour
        hulls = [cv2.convexHull(contour) for contour in contours]
        cv2.drawContours(contour_img, hulls, -1, 255, 1)
        
        # Find contours of the convex hulls
        contours, _ = cv2.findContours(contour_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Draw the final contours on a blank image
        contour_img = np.zeros_like(img_len)
        cv2.drawContours(contour_img, contours, -1, 255, 1)

        # Perform connected component analysis on the final contour image
        _, labels, stats, centroids = cv2.connectedComponentsWithStats(contour_img)
        areas = stats[1:, cv2.CC_STAT_AREA]

        # Get the indices of the two largest areas again
        largest_area_idxs = np.argsort(areas)[-2:] + 1

        return img_bin, centroids, largest_area_idxs, labels

    def lens_localization(self, img_bin, largest_area_idxs, labels, centroids):
        """
        Localizes lens regions in a binary image based on given area indices, labels, and centroids.

        Parameters:
        img_bin (numpy.ndarray): Binary image where lens regions need to be localized.
        largest_area_idxs (list): List of indices representing the largest areas in the image.
        labels (numpy.ndarray): Label matrix where each region in the image is labeled with a unique integer.
        centroids (numpy.ndarray): Array of centroids for each labeled region.

        Returns:
        tuple: A tuple containing:
            - contour_img (numpy.ndarray): Image with the contours of the largest areas drawn.
            - sides (list): List of images, each containing one of the largest areas.
            - centr (list): List of centroids corresponding to the largest areas.
            - pts (list): List of points defining the bounding box and diagonal corners for each of the largest areas.
        """
        # Initialize the contour image and blank image
        contour_img = np.zeros_like(img_bin)
        blank = contour_img.copy()
        
        # Initialize lists to store centroids, sides, and points
        centr = []
        sides = []
        pts = []

        # Iterate over the indices of the largest areas
        for idx in largest_area_idxs:
            # Create a copy of the blank image
            aux = blank.copy()
            
            # Fill the contour image and auxiliary image with the current label
            contour_img[labels == idx] = 255
            aux[labels == idx] = 255
            
            # Append the current auxiliary image and centroid to the lists
            sides.append(aux)
            centr.append(centroids[idx])

            # Find the coordinates of the pixels in the auxiliary image
            y, x = np.where(aux == 255)

            # Calculate the points based on the minimum and maximum coordinates and the centroid
            pt1 = [np.min(x), int(centroids[idx, 1])]    # hozinontal min of len
            pt2 = [np.max(x), int(centroids[idx, 1])]    # hozinontal max of len
            pt3 = [int(centroids[idx, 0]), np.min(y)]    # vertical min of len
            pt4 = [int(centroids[idx, 0]), np.max(y)]    # vertical max of len
            pt5 = [np.min(x), np.min(y[x == np.min(x)])] # min y of min x
            pt6 = [np.max(x), np.min(y[x == np.max(x)])] # min y of max x

            # Append the calculated points to the list
            pts.append([pt1, pt2, pt3, pt4, pt5, pt6])
        
        # Ensure the points are ordered correctly based on the x-coordinate
        if pts[0][0][0] > pts[1][0][0]:
            pts[0], pts[1] = pts[1], pts[0]

        # Ensure the centroids and sides are ordered correctly based on the x-coordinate
        if centr[0][0] > centr[1][0]:
            centr[0], centr[1] = centr[1], centr[0]
            sides[0], sides[1] = sides[1], sides[0]

        # Return the contour image, sides, centroids, and points
        return contour_img, sides, centr, pts

    def find_biggest_distance(self, sides, centr, pts):
        """
        Finds the pairs of points with the maximum Euclidean distance within the segmented areas and adds these points to the pts list.

        Parameters:
        sides (list of numpy.ndarray): List of images, each containing one of the largest areas.
        centr (list): List of centroids corresponding to the largest areas.
        pts (list): List of points defining the bounding box and diagonal corners for each of the largest areas.

        Returns:
        list: Updated list of points with the maximum distance points added for each area.
        """
        def find_max_distance(A, B):
            """Finds the pair of points with the maximum Euclidean distance between two groups of points."""
            max_distance = 0
            max_points = []

            for point in A:
                for point2 in B:
                    distance = euclidean_dist(point, point2)
                    if distance > max_distance:
                        max_distance = distance
                        max_points = [list(point), list(point2)]
            
            return max_points

        def split_coordinates(area, centroid):
            """Splits the coordinates of an area into two groups based on their position relative to the centroid."""
            y_coords, x_coords = np.where(area == 255)
            A = list(zip(x_coords[x_coords < centroid[0]], y_coords[x_coords < centroid[0]]))
            B = list(zip(x_coords[x_coords > centroid[0]], y_coords[x_coords > centroid[0]]))
            return A, B

        for i in range(2):
            A, B = split_coordinates(sides[i], centr[i])
            max_points = find_max_distance(A, B)
            pts[i].extend(max_points)

        return pts

    def adding_additional_points(self, pts, mesh_points, contour_img):

        y,x = np.where(contour_img == 255)

        try:
            (l_cx, l_cy), _ = cv2.minEnclosingCircle(mesh_points[self.LEFT_IRIS])
            (r_cx, r_cy), _ = cv2.minEnclosingCircle(mesh_points[self.RIGHT_IRIS])

            center_left = [int(l_cx), int(l_cy)]
            center_right = [int(r_cx), int(r_cy)]

            ponto_left = [center_left[0], np.min(y[(x == center_left[0]) & (y > center_left[1])])]
            ponto_right = [center_right[0], np.min(y[(x == center_right[0]) & (y > center_right[1])])]

            x = mesh_points[168][0]
            y = int(mesh_points[168][1] + (mesh_points[6][1] - mesh_points[168][1])/2)
            naso = [x, y]

            pts.append([center_left, ponto_left, center_right, ponto_right, naso])

        except Exception as e:
            logger.error('Erro to add points: ', e)

        return pts
