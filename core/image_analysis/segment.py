import cv2
import torch
from ultralytics import SAM
import numpy as np

class Segmentator:
    def __init__(self, model_path: str = 'models/sam2_b.pt'):
        """
        Initializes the Segmentator class with the SAM model.
        
        Args:
            model_path (str): Path to the SAM model. Default is 'models/sam2_b.pt'.
        """
        self.model = SAM(model_path)

    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocesses the input frame to enhance the lens for better segmentation.
        
        Args:
            frame (np.ndarray): Input image/frame to preprocess.
        
        Returns:
            np.ndarray: Preprocessed frame with enhanced lens regions.
        """
        # Apply weighted addition to enhance the lens
        frame = cv2.addWeighted(frame, 2, cv2.GaussianBlur(frame, (0, 0), 10), -0.5, 1)
        # Apply Gaussian blur to smooth the frame
        frame = cv2.GaussianBlur(frame, (0, 0), 10)
        
        return frame
    
    def postprocess(self, frame: np.ndarray, mask1: np.ndarray, mask2: np.ndarray, box1: np.ndarray, box2: np.ndarray) -> np.ndarray:
        """
        Post-processes the masks by removing parts outside the respective bounding boxes.
        
        Args:
            frame (np.ndarray): Original image/frame.
            mask1 (np.ndarray): First mask to be processed.
            mask2 (np.ndarray): Second mask to be processed.
            box1 (np.ndarray): Bounding box coordinates for the first mask.
            box2 (np.ndarray): Bounding box coordinates for the second mask.
        
        Returns:
            np.ndarray: Combined mask after post-processing.
        """
        # Create empty masks with the same shape as the frame
        bbox_mask1 = np.zeros(frame.shape[:2], dtype=np.uint8)
        bbox_mask2 = np.zeros(frame.shape[:2], dtype=np.uint8)

        # Fill the bounding boxes with white (255) in the respective masks
        bbox_mask1[box1[1]:box1[3], box1[0]:box1[2]] = 255
        bbox_mask2[box2[1]:box2[3], box2[0]:box2[2]] = 255
    
        # Apply bitwise AND to keep only the parts of the mask inside the bounding box
        mask1 = cv2.bitwise_and(mask1, mask1, mask=bbox_mask1)
        mask2 = cv2.bitwise_and(mask2, mask2, mask=bbox_mask2)

        # Return the combined mask
        return mask1 + mask2
    
    def predict(self, frame: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        """
        Predicts the segmentation masks for the given frame and bounding boxes.
        
        Args:
            frame (np.ndarray): Input image/frame for segmentation.
            boxes (np.ndarray): Array of bounding boxes to guide the segmentation.
        
        Returns:
            np.ndarray: Combined segmentation mask.
        
        Raises:
            Exception: If an error occurs during segmentation inference.
        """
        try:
            # Extract the first and second bounding boxes
            box1 = boxes[0][None, :][0]
            box2 = boxes[1][None, :][0]
            
            with torch.no_grad():
                # Predict the first mask using SAM model
                mask1 = self.model.predict(frame, bboxes=box1)
                mask1 = mask1[0].masks.data.cpu().numpy()  # Convert tensor to numpy
                
                # Predict the second mask using SAM model
                mask2 = self.model.predict(frame, bboxes=box2)
                mask2 = mask2[0].masks.data.cpu().numpy()  # Convert tensor to numpy

            # Post-process and combine the masks
            mask = self.postprocess(frame, mask1[0] * 255, mask2[0] * 255, box1, box2)
            
            return mask
        
        except Exception as e:
            # Log and raise the exception (logging is commented out)
            # logger.error(f"Error in segmentation inference: {e}")
            raise e
    

if __name__ == '__main__':
    from core.image_analysis.detect import LenDetector

    # Initialize the Segmentator and LenDetector
    segmentator = Segmentator()
    detector = LenDetector()

    # Read the input image
    frame = cv2.imread('/home/vitor/Documents/Cus.AI/M.D.L/len_segmentation/teste_imgs/teste_17-07/4954392295633956944.jpg')
    
    # Detect bounding boxes in the frame
    boxes = detector.predict(frame)

    # Predict segmentation mask using the detected bounding boxes
    mask = segmentator.predict(frame, boxes)


