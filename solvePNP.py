import pandas as pd
import numpy as np
import cv2

# Define the camera matrix and distortion coefficients
camera_matrix = np.array([ [1288.6255, 0, 813.2959],
                            [0, 1290.6448, 819.7536],
                            [0, 0, 1]], dtype=np.float32)

k1, k2, p1, p2, k3 = 0.2172,-0.6233 , -0.0008 ,-0.0004, 0.5242

dist_coeffs = np.array([k1, k2, p1, p2, k3], dtype=np.float32)

fname = f"measures.txt"
with open(fname, 'r') as f:
    lines = f.readlines() 

f_last = 0
count_lamps = 0

object_points_world = []
object_points_image = []
camera_points_world = []
object_points = []
image_points = []
camera_points = []

for line in lines:
    a,b,c,d,e,f = line.split(' ')
    a,b,c,d,e,f  = float(a), float(b), float(c), float(d), e, int(f)
    img_w, img_h = e.split(',')
    img_w, img_h = float(img_w), float(img_h)
    if f_last != f or count_lamps == 4:
        if count_lamps >= 3:
            object_points = np.array(object_points_world, dtype=np.float32)
            image_points = np.array(object_points_image, dtype=np.float32)
            # Solve PNP using the object points, image points, camera matrix, and distortion coefficients
            success, rotation_vector, translation_vector = cv2.solvePnP(object_points, image_points, camera_matrix, None)

            # Convert rotation vector to rotation matrix
            rot_mat, _ = cv2.Rodrigues(rotation_vector)

            # Compute inverse of rotation matrix and translation vector
            rot_mat_inv = np.linalg.inv(rot_mat)
            tvec_inv = np.dot(-rot_mat_inv, translation_vector)

            # Compute camera position in world coordinate system
            cam_pos = tvec_inv.flatten()
            diff = 0
            for i in range(3):
                diff += np.power(cam_pos[i] - camera_points_world[i], 2)
            diff += np.sqrt(diff)
            print(diff)

        count_lamps = 0
        f_last = f
        object_points_world = []
        object_points_image = []
    else:
        count_lamps += 1
    object_points_world.append([a, b, 2.710])
    object_points_image.append([img_w, img_h])
    camera_points_world = [c, d, 0.0]

#object_points = np.array(object_points, dtype=np.float32)
#image_points = np.array(image_points, dtype=np.float32)




