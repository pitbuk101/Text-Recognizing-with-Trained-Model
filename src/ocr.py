import numpy as np
import cv2
import matplotlib.pyplot as plt
import pytesseract
import re

from pytesseract import Output



#helper functions
def plot_gray(image):
    plt.figure(figsize=(16,10))
    return plt.imshow(image, cmap='Greys_r')


def plot_rgb(image):
    plt.figure(figsize=(16,10))
    return plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

file_name = r"C:\Users\tprat\Desktop\Project\OCR\result.png"
image = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE) 
plot_gray(image)



# Text box detection
pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'  # your path may be different
d = pytesseract.image_to_data(image, output_type=Output.DICT)

n_boxes = len(d['level'])
boxes = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
for i in range(n_boxes):
    (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
    boxes = cv2.rectangle(boxes, (x, y), (x + w, y + h), (0, 255, 0), 2)

plot_rgb(boxes)



# Text recognition
extracted_text = pytesseract.image_to_string(image)
print(extracted_text)



# Extract info on Invoice No. and grand total
def find_amounts(text):
    decimal_numbers = re.findall(r'(\d+\s*\.\s*\d{2})\b', text)
    print(decimal_numbers)
    decimal_numbers = [float(re.sub(r'\s+', '', num)) for num in decimal_numbers]
    unique = list(dict.fromkeys(decimal_numbers))
    return unique
def find_invoice_no(text):
    patterns = [
        r'Invoice Number\s*(\d+)',              # all combinations as per the standards can be added
        r'Invoice No :\s*(\d+)',
        r'Invoice No\s+(\d+)',
        r'Invoice Number\s+(\d+)',
        r'Bill No\. :\s*(\d+)',
        r'Qrder No:\s*([A-Za-z0-9]+)',
        r'B1ll Ho\.:([\w\-\/]+)'

    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1)
    return None


amounts = find_amounts(extracted_text)
print(f'All amounts ending with .00 are : {amounts}')
# Grand total is the largest one

print(f"Grand total is the largest one : {max(amounts)}")

# Invoice no. will be after "Invoice No :"  , also you can add all the possibility of writing invoice number to get serced by that.
invoice_no = find_invoice_no(extracted_text)
print(f'Invoice/Bill Number : {invoice_no}')   # perfect solution to this will be NER model