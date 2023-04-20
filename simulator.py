import numpy as np
from matplotlib import pyplot as plt


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

def project_points(points_3d, K, cam_rot, t):
    """
    Projects 3D points onto a 2D image plane using the pinhole camera model.

    Args:
        points_3d: A numpy array of shape (n, 3) containing the 3D coordinates of the points to be projected.
        K: A numpy array of shape (3, 3) containing the camera intrinsic matrix.
        R: A numpy array of shape (1, 3) containing the camera rotation degres.
        t: A numpy array of shape (3, 1) containing the camera translation vector.

    Returns:
        A numpy array of shape (n, 2) containing the 2D image coordinates of the projected points.
    """

    assert points_3d.shape[1] == 3, f"World points need to be of shape (n, 3)"  
    assert K.shape[0] == 3 and K.shape[1] == 3, f"camera intrinsics need to be of shape (3, 3)"  
    assert cam_rot.shape[0] == 3, f"camera rotation need to be a vector of shape (3)" 
    assert t.shape[0] == 3 and t.shape[1] == 1, f"translation points need to be of shape (3, 1)"  

    # transform world coordinates into homegeneous
    points_3d_hom = np.hstack((points_3d, np.ones((points_3d.shape[0], 1))))
    
    # calculate the Rotation Matrix
    theta_x = np.deg2rad(cam_rot[0]) 
    theta_y = np.deg2rad(cam_rot[1]) 
    theta_z = np.deg2rad(cam_rot[2])
    
    Rx = np.array([[1, 0, 0],
               [0, np.cos(theta_x), -np.sin(theta_x)],
               [0, np.sin(theta_x), np.cos(theta_x)]])

    Ry = np.array([[np.cos(theta_y), 0, np.sin(theta_y)],
                [0, 1, 0],
                [-np.sin(theta_y), 0, np.cos(theta_y)]])

    Rz = np.array([[np.cos(theta_z), -np.sin(theta_z), 0],
                [np.sin(theta_z), np.cos(theta_z), 0],
                [0, 0, 1]])

    R = np.dot(Rz, np.dot(Ry, Rx))

    # stack the rotation matrix with translation to obtain a (3,4) matrix

    RT = np.hstack((R, t))


    points_3d_cam = np.dot(RT, points_3d_hom.T)    
    points_2d_hom = np.dot(K, points_3d_cam)

    # go from 3d to 2d in the image plane
    points_2d = (points_2d_hom[:2, :] / points_2d_hom[2, :]).T

    return points_2d




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
            camera_rotation = np.array([0, 0, r])
            translation = np.array([[x], [y], [z]])
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



# generate camera space positions
x_mx_room = 3
y_mx_room = 2.5
size_x = int(x_mx_room / 0.25) + 1
size_y = int(y_mx_room / 0.25) + 1

xv = np.linspace(0,3,size_x)
yv = np.linspace(0, 2, size_y)

camera_positions = []
for xi in xv:
    for yi in yv:
        camera_positions.append([xi, yi, 0])
camera_positions = np.array(camera_positions)

points = simulate(camera_positions, K, image_width, image_height)


write = True
if write:
    write_to_file("measures.txt", points)

