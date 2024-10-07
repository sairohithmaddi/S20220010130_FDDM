import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load images
img1_path = r"C:\Users\sairo\Downloads\img2.png"
img2_path = r"C:\Users\sairo\Downloads\img4.png"

img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

# Harris corner detection function
def harris_corner_detection(image):
    corners = cv2.cornerHarris(image, blockSize=2, ksize=3, k=0.04)
    corners = cv2.dilate(corners, None)
    return corners

corners1 = harris_corner_detection(img1)
corners2 = harris_corner_detection(img2)

# Create color images for drawing corners
img1_color = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
img2_color = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)

# Draw corners in red
img1_color[corners1 > 0.01 * corners1.max()] = [0, 0, 255]  # Red
img2_color[corners2 > 0.01 * corners2.max()] = [0, 0, 255]  # Red

# Show corners detected
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title('Image 1 Corners')
plt.imshow(img1_color)
plt.subplot(1, 2, 2)
plt.title('Image 2 Corners')
plt.imshow(img2_color)
plt.show()


