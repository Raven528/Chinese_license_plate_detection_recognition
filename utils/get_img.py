import cv2
import base64
import logging
import requests
import numpy as np
from io import BytesIO
from typing import Any, Tuple
from urllib.parse import urlparse

def download_url(url: str) -> Tuple[Any, str]:
    if not (url and urlparse(url).scheme):
        error_message = "Invalid or empty URL."
        logging.error(error_message)
        raise ValueError(error_message)

    try:
        response = requests.get(url, timeout=10) 
    except requests.exceptions.Timeout as e:
        error_message = f"Request timed out: {e}"
        logging.error(error_message)
        raise
    except requests.exceptions.RequestException as e:
        error_message = f"Request exception: {e}"
        logging.error(error_message)
        raise
    except Exception as e:
        error_message = f"Unexpected error: {e}"
        logging.error(error_message)
        raise
    
    if response.status_code == 200:
        data = BytesIO(response.content)
        image_array = np.frombuffer(data.getvalue(), dtype=np.uint8)
        try:
            img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            logging.info("Image downloaded and processed successfully.")
            return img
        except Exception as e:
            error_message = f"Failed to process image: {e}"
            logging.error(error_message)
            raise
    elif response.status_code == 403:
        error_message = "Access to the URL is forbidden."
        logging.error(error_message)
        raise PermissionError(error_message)
    elif response.status_code == 404:
        error_message = "URL not found."
        logging.error(error_message)
        raise FileNotFoundError(error_message)
    else:
        error_message = f"Invalid URL with status code: {response.status_code}"
        logging.error(error_message)
        raise ValueError(error_message)

def base64_to_image(base64_string):
    """将 Base64 字符串解码为 OpenCV 图像"""
    # 将 Base64 字符串解码为二进制数据
    image_data = base64.b64decode(base64_string)
    
    # 将二进制数据转换为 NumPy 数组
    image_array = np.frombuffer(image_data, dtype=np.uint8)
    
    # 解码为 OpenCV 图像
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Base64 字符串解码失败")
    
    return image

def reqBody2img(req_body):
    """Process the image based on the provided request body."""
    img_base64 = req_body.get('image', None)
    img_path = req_body.get('img_path', None)
    url = req_body.get('url', None)
    
    img = None
    
    # base64 to image
    if img_base64 is not None:
        img = base64_to_image(img_base64)
        print('using base64 to image')
    # URL to image
    elif url is not None:
        img, errcode = download_url(url)
        print('using url to image')
    # image path to image
    elif img_path is not None:
        img = cv2.imread(img_path)
        print('using image path to image')
    # If no valid input is provided
    if img is None:
        raise ValueError("No valid image source provided.")

    return img