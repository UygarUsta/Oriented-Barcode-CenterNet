"""
A basic script to augment your dataset by applying random rotation for each image
This will read labels from xml/ folder and images from images/ and will generate
an augmented dataset to images_augmented/ with a csv file train.csv
containing x, y, width, height and angle for each car in the image
"""
import os
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import cv2
from tqdm import tqdm

dicts = []
idx = 1

# Function to resize image while preserving aspect ratio and padding
def resize_and_pad(image, target_size=(512, 512)):
    original_height, original_width = image.shape[:2]
    target_width, target_height = target_size

    # Calculate the scaling factor
    scale = min(target_width / original_width, target_height / original_height)
    
    # Calculate new dimensions
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)
    
    # Resize the image
    resized_image = cv2.resize(image, (new_width, new_height))
    
    # Pad the image to the target size
    delta_w = target_width - new_width
    delta_h = target_height - new_height
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)
    padded_image = cv2.copyMakeBorder(resized_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    
    return padded_image, scale, left, top

# Loop over xml files
for label_path in tqdm(sorted(os.listdir("xml"))):
    mytree = ET.parse(os.path.join("xml", label_path))
    root = mytree.getroot()

    target_size = (512,512)

    # Read image with OpenCV
    img = cv2.imread(root.find("path").text)
    
    padded_image, scale, offset_x, offset_y = resize_and_pad(img, target_size)
    img = padded_image

    # Generate random rotation (identity + 9 rotations)
    rand_rotations = np.random.uniform(low=-180, high=180, size=(9,))
    rand_rotations = np.append(rand_rotations, [0.0])

    for rot_angle in rand_rotations:
        image_center = tuple(np.array(img.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, rot_angle, 1.0)

        # Rotate image around image center
        img_rotated = cv2.warpAffine(img, rot_mat, img.shape[1::-1], flags=cv2.INTER_LINEAR)

        # Parse xml file
        for robndbox in root.findall("object/robndbox"):
            x = float(robndbox.find("cx").text) * scale + offset_x
            y = float(robndbox.find("cy").text) * scale + offset_y
            w = float(robndbox.find("w").text) * scale
            h = float(robndbox.find("h").text) * scale
            angle = float(robndbox.find("angle").text)  # between 0 and 2*PI

            # Correct bbox position by applying rotation matrix
            correct = np.dot(rot_mat, np.array([x, y, 1]).reshape(3, 1))

            # It can happen that this random rotation pushes objects out of bounds
            if not 0 <= correct[0] < img.shape[1] or not 0 <= correct[1] < img.shape[0]:
                continue

            # Add rotation angle
            angle = np.pi / 2 - angle + np.pi / 180 * rot_angle

            if angle > np.pi:
                angle -= 2 * np.pi
            elif angle <= -np.pi:
                angle += 2 * np.pi

            # Resulting angle is between -PI and +PI
            assert -np.pi <= angle <= np.pi

            # Convert width of labelImg2 to length of the vehicle
            # Convert height of labelImg2 to "width" of the vehicle
            dicts.append(
                {
                    "name": f"image_{idx:04d}",
                    "img_width": img.shape[1],  # Corrected to get width
                    "img_height": img.shape[0],  # Corrected to get height
                    "x": int(correct[0]),
                    "y": int(correct[1]),
                    "w": f"{h:.2f}",
                    "l": f"{w:.2f}",
                    "angle": angle,
                }
            )

        cv2.imwrite(f"images_augmented/image_{idx:04d}.png", img_rotated)
        idx += 1

    debug = False  # Set to True to debug it
    if debug:
        for car in dicts:
            if car["name"] == f"image_{idx-1:04d}":
                print(car)
                cos_angle = np.cos(car["angle"])
                sin_angle = np.sin(car["angle"])
                rot = np.array([[cos_angle, sin_angle], [-sin_angle, cos_angle]])

                w = float(car["w"])
                l = float(car["l"])

                bottom_right = np.dot(rot, np.array([w / 2, l / 2]).reshape(2, 1)).reshape(2)
                top_right = np.dot(rot, np.array([w / 2, -l / 2]).reshape(2, 1)).reshape(2)
                top_left = np.dot(rot, np.array([-w / 2, -l / 2]).reshape(2, 1)).reshape(2)
                bottom_left = np.dot(rot, np.array([-w / 2, l / 2]).reshape(2, 1)).reshape(2)

                br = (int(car["x"] + bottom_right[0]), int(car["y"] + bottom_right[1]))
                tr = (int(car["x"] + top_right[0]), int(car["y"] + top_right[1]))
                tl = (int(car["x"] + top_left[0]), int(car["y"] + top_left[1]))
                bl = (int(car["x"] + bottom_left[0]), int(car["y"] + bottom_left[1]))

                thickness = 3
                cv2.line(img_rotated, br, tr, (0, 220, 0), thickness)  # Draw front of the vehicle in another color
                cv2.line(img_rotated, br, bl, (220, 220, 0), thickness)
                cv2.line(img_rotated, tl, bl, (220, 220, 0), thickness)
                cv2.line(img_rotated, tl, tr, (220, 220, 0), thickness)

        cv2.imshow("Img", img_rotated)
        k = cv2.waitKey(0)

        if k == 27:
            cv2.destroyAllWindows()
            break

# Save targets to csv file
df = pd.DataFrame(dicts)
df.to_csv("train.csv", index=False)
