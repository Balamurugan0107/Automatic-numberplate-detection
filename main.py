import cv2
import imutils
import numpy as np
from PIL import Image
import pytesseract  # Optional, in case you want to use it later for OCR

# Load the image using the correct path
image_path = "C:/Users/Mohammed Javeed Ali/Dropbox/PC/Downloads/car.jpg"
image = cv2.imread(image_path)

# Check if the image was loaded correctly
if image is None:
    print(f"Error: Could not load image at path '{image_path}'")
    exit(1)

# Resize the image using imutils
image = imutils.resize(image, width=500)

# Convert to grayscale and apply filtering
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.bilateralFilter(gray, 11, 17, 17)

# Edge detection
edged = cv2.Canny(gray, 170, 200)

# Find contours
cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:30]

# Detect the number plate contour
number_plate_cnt = None
for c in cnts:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    if len(approx) == 4:
        number_plate_cnt = approx
        break

# Draw the contour on the original image and extract the number plate region
if number_plate_cnt is not None:
    cv2.drawContours(image, [number_plate_cnt], -1, (0, 255, 0), 3)

    # Create a mask for the number plate
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [number_plate_cnt], -1, 255, -1)
    
    # Extract the number plate region from the image
    x, y, w, h = cv2.boundingRect(number_plate_cnt)
    number_plate = gray[y:y+h, x:x+w]

    # Convert the number plate region to a PIL image for further processing
    pil_img = Image.fromarray(number_plate)

    # Save or display the extracted number plate region (optional)
    cv2.imwrite("extracted_number_plate.jpg", number_plate)
    pil_img.show()

    # You can use your custom OCR method here
    # For example, if using PIL to read text:
    # text = pytesseract.image_to_string(pil_img, config='--psm 8')
    # print(f"Detected Number Plate: {text.strip()}")
else:
    print("Error: Could not detect the number plate contour.")

# Display the result with the number plate highlighted
cv2.imshow("Detected Number Plate", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
