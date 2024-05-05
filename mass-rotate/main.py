import os
import cv2

# Create the output directory if it doesn't exist
if not os.path.exists('output'):
    os.makedirs('output')

# Loop through all files in the current directory
for filename in os.listdir('.'):
    # Check if the file is an image
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
        # Read the image
        img = cv2.imread(filename)
        # Rotate the image
        img = cv2.rotate(img, cv2.ROTATE_180)
        # Save the image in the output directory with the same name
        cv2.imwrite(os.path.join('output', filename), img)