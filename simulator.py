import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
"""
Positions are given in meters
"""
# x y z
point_coordinates = np.array([
    [0.330, 0.485, 2.710],
    [1.523 ,0.485, 2.710],
    [2.713 ,0.485, 2.710],
    [3.908 ,0.492, 2.710],
    [0.335 ,2.080, 2.710],
    [1.528 ,2.087, 2.710],
    [2.713 ,2.081, 2.710],
    [3.908 ,2.085, 2.710]
],dtype=float)

camera_orientation = np.array([90, 90])
camera_position = np.array([0, 1, 0])

image_width = 3264
image_height = 2464
def genPos(point_coordinates, camera_orientation, camera_position, img_size):
    
    """
    image_width, image_height = img_size
    sensor_width = 150 # in mm
    sensor_height = 150 # in mm
    focal_length = 3.04
    fx = focal_length * image_width / sensor_width
    fy = focal_length * image_height / sensor_height
    cx = image_width / 2
    cy = image_height / 2
    A = np.array([[fx, 0, cx],
                    [0, fy, cy],
                    [0, 0, 1]])
    """
    A = np.array([[1288.6255, 0, 813.2959],
                                [0, 1290.6448, 819.7536],
                                [0, 0, 1]])

    n_points = point_coordinates.shape[0]

    p = np.zeros((int(n_points), 4))

    p[:, 0] = point_coordinates[:, 0]
    p[:, 1] = point_coordinates[:, 1]
    p[:, 2] = point_coordinates[:, 2]
    p[:, 3] = 1

    """
    This implementation only as in considerantion rotation around x and y 
    """
    camera_orientation = np.deg2rad(camera_orientation)
    # x rotation
    r1 = np.eye(3) 
    r1[1,1] = np.cos(-camera_orientation[0])
    r1[1,2] = -np.sin(-camera_orientation[0])
    r1[2,1] = np.sin(-camera_orientation[0])
    r1[2,2] = np.cos(-camera_orientation[0])

    # y rotation 
    r2 = np.eye(3)
    r2[0,0] = np.cos(-camera_orientation[1])
    r2[0,2] = np.sin(-camera_orientation[1])
    r2[2,0] = -np.sin(-camera_orientation[1])
    r2[2,2] = np.cos(-camera_orientation[1])
    # zyxt

    R = r2 @ r1
    
    T = np.array([-camera_position[0], -camera_position[1], -camera_position[2]]).T

    B = np.c_[R, T]

    P = np.zeros((n_points, 3))

    for i in range(n_points):
        P[i,:] = np.dot(A, np.dot(B, p[i,:].T))

    P[:,0] = P[:,0] / P[:,2] 
    P[:,1] = P[:,1] / P[:,2] 

    x = P[:,0]
    y = P[:,1]

    return x, y

x_mx_room = 3
y_mx_room = 2.5
size_x = int(x_mx_room / 0.1) + 1
size_y = int(y_mx_room / 0.1) + 1

xv = np.linspace(0,3,size_x)
yv = np.linspace(0, 2, size_y)
r = np.linspace(0,360, 37) # rotation of 10 degrees

# TODO: clean this part
data = {}
fname = f"measures"
with open(fname, 'w') as f:
    for xc in xv:
        xc = round(xc, 2)
        for yc in yv:
            yc = round(yc, 2)
            for rot in r:
                camera_orientation = np.array([90, rot])
                camera_position = np.array([xc, yc, 0])
                x, y = genPos(point_coordinates, camera_orientation, camera_position, (image_width, image_height))
                cond1x = x > 0
                cond2x = x < image_width
                condx  =  cond1x & cond2x
                cond1y = y > 0
                cond2y = y < image_height
                condy = cond1y & cond2y
                x_i = np.where(condx, x, 0)
                y_i = np.where(condy, y, 0)
                x_i = np.where(x_i > 0)
                y_i = np.where(y_i > 0)
                common_val = np.intersect1d(x_i, y_i)
                k = f"{xc},{yc},{rot}"
                if common_val.shape[0] == 0: continue
                if rot != 0: rot = rot/360 
                f.write(f"- {xc} {yc} {rot/360}\n")
                for c in common_val:
                    f.write(f"{c} {round(x[c]/image_width, 3)} {round(y[c]/image_height,3)}\n")


"""
xv = np.linspace(0,3,3*25)
yv = np.linspace(0, 2, 10)
r = np.linspace(0,360,10)

fig, ax = plt.subplots()
s = ax.scatter([], [])
ax.set_xlim(-image_width, image_width)
ax.set_ylim(-image_height, image_height)

def update(i):

    s.set_offsets(np.column_stack([xs[i], ys[i]]))


xs = []
ys = []
for xc in xv:
    for yc in yv:
        for rot in r:
            camera_position = np.array([xc, 0.0, yc])
            camera_orientation = np.array([90, rot])
            x, y = genPos(point_coordinates, camera_orientation, camera_position, (image_width, image_height))
            if x.shape[0] == 0: continue
            xs.append(x)
            ys.append(y)
            

ani = FuncAnimation(fig, update, frames=5, interval=1000)

plt.show()

"""

"""
import cv2
import numpy as np

# Define the 3D points of the object to be detected
object_points = np.array([[0,0,0], [0,1,0], [1,1,0], [1,0,0]], dtype=np.float32)

# Load the image and extract the 2D points of the object
image = cv2.imread('image.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
ret, corners = cv2.findChessboardCorners(gray, (2,2), None)
image_points = corners.reshape(-1,2)

# Define the camera matrix and distortion coefficients
camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
dist_coeffs = np.array([k1, k2, p1, p2, k3], dtype=np.float32)

# Solve PNP using the object points, image points, camera matrix, and distortion coefficients
success, rotation_vector, translation_vector = cv2.solvePnP(object_points, image_points, camera_matrix, dist_coeffs)

# Print the results
print("Rotation vector:\n", rotation_vector)
print("Translation vector:\n
"""

"""
import cv2
import numpy as np

# 3D coordinates of the points in the world coordinate system
object_points = np.array([[0,0,0], [0,1,0], [1,1,0], [1,0,0]], dtype=np.float32)

# 2D coordinates of the points in the camera image
image_points = np.array([[10,10], [20,30], [30,30], [30,10]], dtype=np.float32)

# Intrinsic camera matrix (focal length, principal point)
K = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=np.float32)

# Distortion coefficients (k1, k2, p1, p2, k3)
dist_coeffs = np.array([0, 0, 0, 0, 0], dtype=np.float32)

# Estimate camera pose using solvePnP function
success, rvec, tvec = cv2.solvePnP(object_points, image_points, K, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

# Print the rotation and translation vectors
print("Rotation vector:\n", rvec)
print("Translation vector:\n", tvec)
"""

# solving the perspective n point projection problem - using the iterative solver
#(success, rotation_vector, translation_vector) = cv2.solvePnP(world_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

# converting the rotation vector to a rotation matrix
#rot_mat, jacobian = cv2.Rodrigues(rotation_vector)
#       Estimate the object pose
#    object_position = -np.matrix(rotation_matrix).T * np.matrix(translation_vector)
#    object_orientation = cv2.decomposeProjectionMatrix(np.hstack((rotation_matrix, translation_vector)))[6]

camera_intrinsic = np.array([[1288.6255, 0, 813.2959],
                             [0, 1290.6448, 819.7536],
                             [0, 0, 1]])


distorcion_coefficients = np.array([0.2172, -0.6233, -0.0008, -0.0004, 0.5242])

focal_length = 3.04 # mm
img_res = np.array([3264, 2464])

