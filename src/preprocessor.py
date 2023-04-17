import numpy as np
import cv2
import matplotlib.pyplot as plt

from skimage.filters import threshold_local     # Local because of non-uniformity on image
from PIL import Image


# # Sample file out of the dataset

# Load the image
img = cv2.imread(r"C:\Users\tprat\Desktop\Project\OCR\Assignment-20230401T120858Z-001\Assignment\image10.jpg")
rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Display binary image
plt.imshow(rgb_img)
plt.show()


# Scale the image
scale_percent = 200
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)
resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

# Increase contrast
gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
contrast_img = clahe.apply(gray)

# Binarize image
threshold_value = 100
binary_img = cv2.threshold(contrast_img, threshold_value, 255, cv2.THRESH_BINARY)[1]

# Noise removal
kernel = np.ones((3,3),np.uint8)
opening = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, kernel, iterations=2)
closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=2)

# Skew correction
coords = np.column_stack(np.where(closing > 0))
angle = cv2.minAreaRect(coords)[-1]
if angle < -45:
    angle = -(90 + angle)
else:
    angle = -angle
(h, w) = closing.shape[:2]
center = (w // 2, h // 2)
M = cv2.getRotationMatrix2D(center, angle, 1.0)
rotated = cv2.warpAffine(closing, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

# Save skew corrected image as PNG
cv2.imwrite("Binarize.png", binary_img)

# # Display the results

cv2.imshow("Binarize", binary_img)

cv2.waitKey(0)
cv2.destroyAllWindows()
