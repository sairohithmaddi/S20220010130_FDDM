import cv2
import numpy as np
import matplotlib.pyplot as plt

img1_path = r"C:\Users\sairo\OneDrive\Desktop\image2.png"
img2_path = r"C:\Users\sairo\OneDrive\Desktop\image4.png"
img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)
sift = cv2.SIFT_create()
keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
keypoints2, descriptors2 = sift.detectAndCompute(img2, None)
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
matches = bf.knnMatch(descriptors1, descriptors2, k=2)
good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)
matching_result = cv2.drawMatches(img1, keypoints1, img2, keypoints2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
plt.figure(figsize=(10, 5))
plt.title('Feature Matches')
plt.imshow(matching_result)
plt.axis('off')
plt.show()
print(f'Number of good matches: {len(good_matches)}')
