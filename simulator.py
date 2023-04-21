import numpy as np
from matplotlib import pyplot as plt
import cv2

points_3d = np.array([
    [0.330, 0.485, 2.710],
    [1.523 ,0.485, 2.710],
    [2.713 ,0.485, 2.710],
    [3.908 ,0.492, 2.710],
    [0.335 ,2.080, 2.710],
    [1.528 ,2.087, 2.710],
    [2.713 ,2.081, 2.710],
    [3.908 ,2.085, 2.710]
],dtype=float)

K = np.array([[1288.6255, 0, 813.2959],
              [0, 1290.6448, 819.7536],
              [0, 0, 1]])


image_width = 3264
image_height = 2464

def project_points(points_3d, K, rvec, tvec):
    """
    Projects 3D points onto a 2D image plane using the pinhole camera model.

    Args:
        points_3d: A numpy array of shape (n, 3) containing the 3D coordinates of the points to be projected.
        K: A numpy array of shape (3, 3) containing the camera intrinsic matrix.
        R: A numpy array of shape (3, 1) containing the camera rotation degres.
        t: A numpy array of shape (3, 1) containing the camera translation vector.

    Returns:
        A numpy array of shape (n, 2) containing the 2D image coordinates of the projected points.
    """

    assert points_3d.shape[1] == 3, f"World points need to be of shape (n, 3)"  
    assert K.shape[0] == 3 and K.shape[1] == 3, f"camera intrinsics need to be of shape (3, 3)"  
    assert rvec.shape[0] == 3, f"camera rotation need to be a vector of shape (3)" 
    assert tvec.shape[0] == 3, f"translation points need to be a vector of shape (3)"  

    image_points, _ = cv2.projectPoints(points_3d, rvec, tvec, K, None)

    return image_points.reshape(image_points.shape[0], -1)



def simulate(camera_positions, K, image_width, image_height):
    """
    Simulates the generation of data points in a image from a known camera position

    Args:
    camera_positions: A numpy array (n,3) that contains the x, y, z camera positions in a 3d space
    image_width: image size width in pixels
    image_height: image size height in pixels
    Returns:
        A numpy array
    """
    points = []

    # loop all the camera positons
    for c_p in camera_positions:
        x, y, z = c_p[0], c_p[1], c_p[2]

        # rotate the camera 360 degrees
        for r in range(0, 360, 10):
            camera_rotation = np.array([0, 0, np.deg2rad(r)])
            translation = np.array([x, y, z])
            points_2d = project_points(points_3d, K, camera_rotation, translation)
            # verify if the points are inside the image and save to a list
            for i in range(points_3d.shape[0]):
                x_img, y_img = points_2d[i][0], points_2d[i][1]
                if x_img >= 0 and x_img <= image_width and y_img >= 0 and y_img <= image_height:
                    points.append(
                        [round(points_3d[i][0],3), round(points_3d[i][1],3), round(points_3d[i][2],3), 
                         round(x, 3), round(y, 3), round(z, 3), 
                         round(x_img,3), round(y_img, 3), 
                         round(r,3)]
                        )
    return points

def write_to_file(fname, data):
    with open(fname, 'w') as f:
        for p in data:
            f.write(f"{p[0]} {p[1]} {p[3]} {p[4]} {p[6]},{p[7]} {p[8]}\n")


def generate_camera_space_positions():
    # generate camera space positions
    x_mx_room = 3
    y_mx_room = 2.5
    size_x = int(x_mx_room / 0.25) + 1
    size_y = int(y_mx_room / 0.25) + 1

    xv = np.linspace(0.25, 3, size_x)
    yv = np.linspace(0.25, 2, size_y)

    camera_positions = []
    for xi in xv:
        for yi in yv:
            camera_positions.append([xi, yi, 0])
    camera_positions = np.array(camera_positions)
    return camera_positions

camera_positions = generate_camera_space_positions()

points = simulate(camera_positions, K, image_width, image_height)

if __name__ == "__main__":
    write = True
    if write:
        write_to_file("measures.txt", points)

