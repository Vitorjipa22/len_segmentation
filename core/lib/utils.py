import gc
import cv2
import torch
import psutil
import numpy as np


def release_memory(self):
    print('Releasing memory...')
    memory_stats('Memory stats before releasing memory:')
    torch.cuda.synchronize()  # Wait for all CUDA operations to complete
    del self.model
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.synchronize()  # Wait for all CUDA operations to complete
    memory_stats('Memory stats after releasing memory:')

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
