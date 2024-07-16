import gc
import cv2
import torch
import numpy as np
import mediapipe as mp
import psutil

from ultralytics import YOLO
from segment_anything import sam_model_registry, SamPredictor

RIGHT_IRIS = [474, 475, 476, 477]
LEFT_IRIS = [469, 470, 471, 472]


def hex(r, g, b): return '#{:02x}{:02x}{:02x}'.format(r, g, b)



def memory_stats(description = "Memory stats:"):
    # System RAM memory
    memory = psutil.virtual_memory()
    print(f"{description}:")
    print(f"System RAM - Total: {memory.total / (1024**3):.2f} GB")
    print(f"System RAM - Used: {memory.used / (1024**3):.2f} GB")
    print(f"System RAM - Available: {memory.available / (1024**3):.2f} GB")
    print(f"System RAM - Usage Percentage: {memory.percent}%")

    # GPU memory
    if torch.cuda.is_available():
        torch.cuda.synchronize()  # Wait for all kernels in all streams on a CUDA device to complete
        gpu_memory = torch.cuda.memory_stats(device=0)  # Get memory stats for the CUDA device
        print(f"GPU - Allocated: {gpu_memory['allocated_bytes.all.current'] / (1024**3):.2f} GB")
        print(f"GPU - Cached: {gpu_memory['reserved_bytes.all.current'] / (1024**3):.2f} GB")
    print()

def euclidean_dist(A, B):
    A = np.array(A)
    B = np.array(B)
    dist = np.linalg.norm(A - B)

    return dist

def pre_process(img):
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

def resize_image(image, scale_percent):
    """
    Redimensiona uma imagem mantendo as proporções de acordo com a porcentagem de redução fornecida.

    Args:
    image (numpy array): Imagem de entrada.
    scale_percent (float): Porcentagem de redução (0 a 100).

    Returns:
    numpy array: Imagem redimensionada.
    """
    # Verifica se a porcentagem é válida
    if scale_percent <= 0 or scale_percent > 100:
        raise ValueError("A porcentagem de redução deve estar entre 0 e 100.")
    
    # Obtém as dimensões da imagem original
    original_height, original_width = image.shape[:2]
    
    # Calcula as novas dimensões
    new_width = int(original_width * (scale_percent / 100))
    new_height = int(original_height * (scale_percent / 100))
    
    # Redimensiona a imagem mantendo as proporções
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
    return resized_image

def lens_localization(img_bin, largest_area_idxs, labels, centroids):
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

def find_biggest_distance(sides, centr, pts):
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

def getting_mesh_points(frame):
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
            print('nenhuma face foi encontrada')

    return mesh_points

def adding_additional_points(pts, mesh_points, contour_img):

    y,x = np.where(contour_img == 255)

    try:
        (l_cx, l_cy), _ = cv2.minEnclosingCircle(mesh_points[LEFT_IRIS])
        (r_cx, r_cy), _ = cv2.minEnclosingCircle(mesh_points[RIGHT_IRIS])

        center_left = [int(l_cx), int(l_cy)]
        center_right = [int(r_cx), int(r_cy)]

        ponto_left = [center_left[0], np.min(y[(x == center_left[0]) & (y > center_left[1])])]
        ponto_right = [center_right[0], np.min(y[(x == center_right[0]) & (y > center_right[1])])]

        x = mesh_points[168][0]
        y = int(mesh_points[168][1] + (mesh_points[6][1] - mesh_points[168][1])/2)
        naso = [x, y]

        pts.append([center_left, ponto_left, center_right, ponto_right, naso])

    except Exception as e:
        print('error to add aditional points: ', e)

    return pts

class Len_calculator:
    def __init__(self, segment_type='vit_h', checkpoint_path='models/sam_vit_h_4b8939.pth', yolo_model_path='models/best-Copy1.pt'):
        memory_stats('Memory stats before initializing the process')
        self.segmentator = Segmentator(segment_type, checkpoint_path)
        self.len_detector = LenDetector(yolo_model_path)
        memory_stats('memory stas after instance the process')

    def predict(self, frame):
        try:
            # frame = resize_image(frame, 80)
            boxes = self.len_detector.predict(frame)
            masks = self.segmentator.predict(frame, boxes)
            masks = masks[0]
            masks = 255 * masks

            return masks
        
        except Exception as e:
            # self.len_detector.release_memory()
            # self.segmentator.release_memory()
            memory_stats('Memory stats before releasing memory')
            torch.cuda.synchronize()  # Wait for all CUDA operations to complete
            del self.len_detector
            self.segmentator = None
            del self.segmentator
            torch.cuda.empty_cache()
            gc.collect()
            torch.cuda.synchronize()  # Wait for all CUDA operations to complete
            memory_stats('Memory stats after releasing memory')
            print(f"An error occurred: {e}")
            raise

class Segmentator:
    def __init__(self, model_type, checkpoint_path):
        self.checkpoint_path = checkpoint_path
        self.model = sam_model_registry[model_type](checkpoint=self.checkpoint_path)
        self.model.to(device=torch.device('cuda:0'))
    
    def predict(self, frame, boxes):
        try:
            print('Segmentando lentes...')
            memory_stats()
            # Convert the frame to RGB
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mask_predictor = SamPredictor(self.model)
            mask_predictor.set_image(image_rgb)
            
            # Predict masks for the two boxes
            with torch.no_grad():
                mask1, _, _ = mask_predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=boxes[0][None, :],
                    multimask_output=False,
                )

                mask2, _, _ = mask_predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=boxes[1][None, :],
                    multimask_output=False,
                )

            # Combine the masks
            mask = mask1 + mask2

            print('Lentes segmentadas com sucesso!')
            self.release_memory()
            
            return mask
        
        except Exception as e:
            self.release_memory()
            print(f"An error occurred: {e}")
            raise
    
    def release_memory(self):
        print('Releasing memory...')
        memory_stats('Memory stats before releasing memory:')
        torch.cuda.synchronize()  # Wait for all CUDA operations to complete
        del self.model
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.synchronize()  # Wait for all CUDA operations to complete
        memory_stats('Memory stats after releasing memory:')

class LenDetector:
    def __init__(self, yolo_model_path):
        self.model = YOLO(yolo_model_path)
        
    
    def predict(self, image):
        try:
            print('Detectando lentes...')
            memory_stats()
            results = self.model.predict(image, save=True)
            boxes = [result.boxes for result in results]
            boxes = boxes[0].xyxy.tolist()

            assert len(boxes) == 2, "The model should return exactly 2 boxes"

            box1 = np.array([int(value) for value in boxes[0]])
            box2 = np.array([int(value) for value in boxes[1]])

            boxes = [box1, box2]

            print('Lentes detectadas com sucesso!')
            self.release_memory()

            return boxes
    
        except Exception as e:
            self.release_memory()
            print(f"An error occurred: {e}")
            raise

    def release_memory(self):
        print('Releasing memory...')
        memory_stats('Memory stats before releasing memory:')
        torch.cuda.synchronize()  # Wait for all CUDA operations to complete
        del self.model
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.synchronize()  # Wait for all CUDA operations to complete
        memory_stats('Memory stats after releasing memory:')