import os
import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector

width, height = 1280, 720
folderpath = 'present'

# Initialize webcam
cap = cv2.VideoCapture(0)
cap.set(3, width)
cap.set(4, height)

# Check if folder exists
if not os.path.exists(folderpath):
    print(f"Folder '{folderpath}' does not exist.")
else:
    pathImages = sorted(os.listdir(folderpath), key=len)
    if not pathImages:
        print(f"No images found in '{folderpath}'.")
    else:
        print(f"Slides found: {pathImages}")

slides = [cv2.imread(os.path.join(folderpath, img)) for img in pathImages]

# Variables
imgNumber = 0
hs, ws = int(120), int(213)
gestureThreshold = 400  # Increased threshold
buttonpress = False
counter = 0
buttonDelay = 15  # Cooldown period after a button press
annotations = [[]]
annotationsnumber = -1
annotationstart = False

# Hand detector
detector = HandDetector(detectionCon=0.7, maxHands=1)

while True:
    print("Reading webcam feed...")
    success, img = cap.read()
    
    if not success:
        print("Failed to access the webcam.")
        break
    
    img = cv2.flip(img, 1)

    # Get the current slide
    imgCurrent = slides[imgNumber].copy()

    # Detect hands
    hands, img = detector.findHands(img)
    print(f"Hands detected: {len(hands)}")

    # Draw gesture threshold line
    cv2.line(img, (0, gestureThreshold), (width, gestureThreshold), (0, 255, 0), 10)

    # Process hand gestures
    if hands and not buttonpress:
        hand = hands[0]
        fingers = detector.fingersUp(hand)
        print(f"Fingers up: {fingers}")

        if len(fingers) == 5:  # Ensure there are 5 values in the fingers list
            cx, cy = hand['center']

            if cy <= gestureThreshold:
                if fingers == [1, 0, 0, 0, 0] and imgNumber > 0:  # Left swipe
                    print('Left gesture detected')
                    imgNumber -= 1
                    buttonpress = True  # Set button press to True to avoid re-triggering
                    counter = 0  # Reset counter after button press
                elif fingers == [0, 0, 0, 0, 1] and imgNumber < len(slides) - 1:  # Right swipe
                    print('Right gesture detected')
                    imgNumber += 1
                    buttonpress = True  # Set button press to True to avoid re-triggering
                    counter = 0  # Reset counter after button press

            # Annotation Drawing
            if fingers == [0, 1, 0, 0, 0]:  # Only index finger up
                if not annotationstart:
                    annotationstart = True
                    annotationsnumber += 1
                    annotations.append([])  # Create a new annotation list
                cv2.circle(imgCurrent, (cx, cy), 12, (0, 0, 255), cv2.FILLED)  # Draw circle
                annotations[annotationsnumber].append((cx, cy))  # Add point to annotations

            elif fingers == [0, 1, 1, 0, 0]:  # Pointer gesture (index and middle fingers up)
                cv2.circle(imgCurrent, (cx, cy), 12, (0, 255, 0), cv2.FILLED)  # Draw pointer circle
                print("Pointer activated at:", (cx, cy))  # Debug statement for pointer activation

            else:
                annotationstart = False  # Reset when finger is not pointing

            # Erase last annotation with [0, 1, 1, 1, 0] (pinky and ring finger up)
            if fingers == [0, 1, 1, 1, 0]:  
                if annotations:  # Check if there are annotations to remove
                    annotations.pop()  # Remove last annotation
                    annotationsnumber -= 1  # Decrease annotation index
                    if annotationsnumber < -1:  # Reset to -1 if all annotations are erased
                        annotationsnumber = -1
                    buttonpress = True  # Prevent re-triggering
                    print("Last annotation erased.")  # Debug statement

    # Button press cooldown logic
    if buttonpress:
        counter += 1
        if counter > buttonDelay:  # Keep button delay logic
            buttonpress = False  # Reset button press

    # Draw all annotations on the current slide
    for i in range(len(annotations)):
        if len(annotations[i]) > 1:  # Only draw lines if there are at least 2 points
            for j in range(len(annotations[i]) - 1):
                cv2.line(imgCurrent, annotations[i][j], annotations[i][j + 1], (0, 0, 200), 12)

    # Display the images
    imgSmall = cv2.resize(img, (ws, hs))
    h, w, _ = imgCurrent.shape
    imgCurrent[0:hs, w - ws:w] = imgSmall
    cv2.imshow("Image", img)
    cv2.imshow("Slides", imgCurrent)

    # Exit condition
    key = cv2.waitKey(1)
    if key == ord('q'):
        print("Exiting...")
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
