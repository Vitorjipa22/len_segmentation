import numpy as np
from ultralytics import YOLO
from core.lib.logger import logger
import cv2

class LenDetector:
    def __init__(self, model_path: str = 'models/best-Copy1.pt'):
        """
        Initializes the LenDetector class with the YOLO model.
        
        Args:
            model_path (str): Path to the YOLO model. Default is 'models/best-Copy1.pt'.
        """
        self.model = YOLO(model_path)
    
    def predict(self, image: np.ndarray) -> list[np.ndarray]:
        """
        Runs the YOLO model on the provided image to detect bounding boxes.
        
        Args:
            image (np.ndarray): Input image for object detection.
        
        Returns:
            list[np.ndarray]: A list containing two bounding boxes as numpy arrays.
        
        Raises:
            AssertionError: If the model does not return exactly 2 bounding boxes.
            Exception: If there is an error during the detection inference.
        """
        try:
            logger.info("Running detection model")
            
            # Run the detection model
            results = self.model.predict(image, save=True, project='runs/detect', name='result', exist_ok=True)
            
            # Extract bounding boxes from the results
            boxes = [result.boxes for result in results]
            boxes = boxes[0].xyxy.tolist()

            # Ensure that exactly two bounding boxes are returned
            assert len(boxes) == 2, "The model should return exactly 2 boxes"

            # Convert bounding box coordinates to numpy arrays
            box1 = np.array([int(value) for value in boxes[0]])
            box2 = np.array([int(value) for value in boxes[1]])

            return [box1, box2]
    
        except Exception as e:
            logger.error(f"Error in detecting inference: {e}")

if __name__ == '__main__':
    # Initialize LenDetector
    len_detector = LenDetector()

    # Load an image
    frame = cv2.imread('/home/vitor/Documents/Cus.AI/M.D.L/len_segmentation/teste_imgs/teste_17-07/4954392295633956944.jpg')

    # Predict bounding boxes in the image
    boxes = len_detector.predict(frame)

    # Print the resulting bounding boxes
    print(boxes)
