# AI-and-Robotics
A collection of my robotics and AI projects focused on innovation, clean energy, and automation.
# My Robotics Projects

This repository contains my personal robotics projects focusing on AI, robotics, and clean energy.

## Projects:
1. **Robot Arm** - A project to create an AI-controlled robot arm.
2. **Autonomous Car** - A small autonomous vehicle project using sensors.
import cv2
import pytesseract

# Load the image
image = cv2.imread('vehicle_image.jpg')

# Preprocess the image
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)
_, thresh = cv2.threshold(blur, 100, 255, cv2.THRESH_BINARY)

# Detect the license plate region (assume a pre-trained model or OpenCV Haar cascade)
plate_cascade = cv2.CascadeClassifier('haarcascade_russian_plate_number.xml')
plates = plate_cascade.detectMultiScale(thresh, scaleFactor=1.1, minNeighbors=5)

for (x, y, w, h) in plates:
    # Crop the license plate region
    plate = image[y:y+h, x:x+w]
    
    # OCR using Tesseract
    text = pytesseract.image_to_string(plate, config='--psm 8')
    print("Detected Plate Number:", text)

    # Draw a rectangle around the license plate
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Show the image with detected plates
cv2.imshow('License Plate Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()