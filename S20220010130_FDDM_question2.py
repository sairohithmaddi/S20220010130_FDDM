import cv2
import numpy as np
import matplotlib.pyplot as plt

img1_path = r"C:\Users\sairo\OneDrive\Desktop\image2.png"
img2_path = r"C:\Users\sairo\OneDrive\Desktop\image4.png"

img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

if img1 is None:
    print("Error: Unable to load image1.")
if img2 is None:
    print("Error: Unable to load image2.")
sift = cv2.SIFT_create()
keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
keypoints2, descriptors2 = sift.detectAndCompute(img2, None)

img1_sift = cv2.drawKeypoints(img1, keypoints1, None)
img2_sift = cv2.drawKeypoints(img2, keypoints2, None)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title('Image 1 SIFT Keypoints')
plt.imshow(img1_sift, cmap='gray')
plt.subplot(1, 2, 2)
plt.title('Image 2 SIFT Keypoints')
plt.imshow(img2_sift, cmap='gray')
plt.show()
