import pybullet as p
import os
import numpy as np
import cv2
import trimesh
import os
import logging

MODE_DEBUG = False

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG if MODE_DEBUG else logging.INFO)

# Connect to PyBullet
p.connect(p.GUI if MODE_DEBUG else p.DIRECT)

# Set the path to the PLY file
brick_name = "brick4x2"


labels_dir = os.path.join(os.path.dirname(__file__), brick_name, "labels")
os.makedirs(labels_dir, exist_ok=True)
img_dir = os.path.join(os.path.dirname(__file__), brick_name, "JPEGImages")
os.makedirs(img_dir, exist_ok=True)
mask_dir = os.path.join(os.path.dirname(__file__), brick_name, "mask")
os.makedirs(mask_dir, exist_ok=True)


# Camera limits make sure that perspectives are unique
camera_limits = {
    "brick2x2": ((0, 0.5), (0, 0.5), (0.1, 0.5)),
    "brick4x2": ((-0.5, 0.5), (0, 0.5), (0.1, 0.5)),
}

camera_limits = camera_limits[brick_name]

mesh_path = os.path.join(os.path.dirname(__file__), brick_name, brick_name + ".urdf")
brick_id = p.loadURDF(mesh_path, basePosition=[0, 0, 0], globalScaling=1)
p.changeVisualShape(brick_id, -1, rgbaColor=[1, 0, 0, .2])  # Set the color to red

mesh = trimesh.load(mesh_path.replace(".urdf", ".obj"))

aabb_min, aabb_max = mesh.bounding_box.bounds

center = (aabb_min + aabb_max) / 2

vertices = np.array([
    [center[0], center[1], center[2], 1],
    [aabb_min[0], aabb_min[1], aabb_min[2], 1],
    [aabb_min[0], aabb_min[1], aabb_max[2], 1],
    [aabb_min[0], aabb_max[1], aabb_min[2], 1],
    [aabb_min[0], aabb_max[1], aabb_max[2], 1],
    [aabb_max[0], aabb_min[1], aabb_min[2], 1],
    [aabb_max[0], aabb_min[1], aabb_max[2], 1],
    [aabb_max[0], aabb_max[1], aabb_min[2], 1],
    [aabb_max[0], aabb_max[1], aabb_max[2], 1],
])

p.addUserDebugPoints(vertices[:, :3], pointColorsRGB=[[1, 0, 0]] * len(vertices), pointSize=10)

# Camera parameters
w, h = 1280, 720
w, h = 640, 480
focal_x = 949.8506622
focal_y = 949.70506462
x_offset = 627.84604663
y_offset = 354.06980147

projection_matrix = [1.484141659687981, 0.0, 0.0, 0.0, 0.0, 2.638069623943646, 0.0, 0.0, 0.018990552138761174, -0.016472773698662655, -1.02020202020202, -1.0, 0.0, 0.0, -0.020202020202020204, 0.0]

projection_matrix_tf = np.array(projection_matrix).reshape(4, 4)

total_samples = 10000
sample_id = 0

from tqdm import tqdm

progress_bar = tqdm(total=total_samples, desc="Generating Data")

# Simulation loop
while sample_id < total_samples:
    p.stepSimulation()

    # Set the camera position to ensure the brick at (0,0,0) is always in view
    camera_position = [np.random.uniform(*camera_limits[0]), np.random.uniform(*camera_limits[1]), np.random.uniform(*camera_limits[2])]
    camera_target = [np.random.uniform(-0.1, 0.1), np.random.uniform(-0.1, 0), 0]  # Keep the z-coordinate fixed to 0 to ensure the brick is in view

    # camera_position = [0.1, 0, .1]
    # camera_target = [0, 0, 0]
    
    camera_up = np.random.uniform(low=[-1, -1, -1], high=[1, 1, 1])
    camera_up /= np.linalg.norm(camera_up)  # Normalize the camera_up vector
    # camera_up = [-1, 0, 0]      # Adjust the up direction to maintain a proper view

    view_matrix = p.computeViewMatrix(camera_position, camera_target, camera_up)
    view_matrix_tf = np.array(view_matrix).reshape(4, 4).T

    # Take a picture
    width, height, rgb, _, seg = p.getCameraImage(width=w, height=h, viewMatrix=view_matrix, projectionMatrix=projection_matrix)

    logger.debug("View Matrix")
    logger.debug(view_matrix_tf)
    logger.debug("Projection Matrix")
    logger.debug(projection_matrix_tf)

    logger.debug("View Space Projection")
    view_space_projection = view_matrix_tf @ vertices.T
    logger.debug(view_space_projection)

    logger.debug("Clip Space Projection")
    clip_space_projection = projection_matrix_tf.T @ view_space_projection
    logger.debug(clip_space_projection)

    logger.debug("Normalized Projection")
    normalized_projection = clip_space_projection / clip_space_projection[-1:]
    logger.debug(normalized_projection)

    coordinates = normalized_projection[:2].T
    if np.any(coordinates < -1) or np.any(coordinates > 1):
        logger.debug("Coordinates out of bounds")
        
        # rgb_image = np.array(rgb)
        # rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)  # Convert from RGB to BGR 
        # cv2.imshow("Camera Image", rgb_image)
        # key = cv2.waitKey(0)  # Wait for a short period to display the image

        continue

    logger.debug("Coordinates")
    logger.debug(coordinates)

    x_range = coordinates[0].max() - coordinates[0].min()
    y_range = coordinates[1].max() - coordinates[1].min()

    class_label = 0

    label = f'{class_label} {" ".join(f"{value:.6f}" for value in coordinates.flatten().tolist())} {x_range:.6f} {y_range:.6f} {focal_x:.6f} {focal_y:.6f} {width} {height} {x_offset:.6f} {y_offset:.6f} {width} {height}'

    sample_file_name = f"{sample_id:06}"

    file_path = os.path.join(labels_dir, f"{sample_file_name}.txt")

    with open(file_path, "w") as f:
        f.write(label)

    img_path = os.path.join(img_dir, f"{sample_file_name}.jpg")
    rgb_image = np.array(rgb)
    rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR) 
    cv2.imwrite(img_path, rgb_image)

    mask_path = os.path.join(mask_dir, f"{sample_file_name}.png")
    seg += 1
    seg *= 255
    cv2.imwrite(mask_path, seg)

    sample_id += 1
    progress_bar.update(1)
    if MODE_DEBUG:
        logger.debug("Pixel Coordinates")
        # Convert normalized coordinates to pixel coordinates
        pixels_x = normalized_projection[0] * w/2 + w/2
        pixels_y = -normalized_projection[1] * h/2 + h/2
        
        pixels_x = pixels_x.astype(int)
        pixels_y = pixels_y.astype(int)

        logger.debug(pixels_x)
        logger.debug(pixels_y)

        # logger.debug(corners_projected.shape)
        # Convert the RGB image from PyBullet to a format suitable for OpenCV
        rgb_image = np.array(rgb)
        rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)  # Convert from RGB to BGR format

        for pp, (pixel_x, pixel_y) in enumerate(zip(pixels_x, pixels_y)):
            cv2.circle(rgb_image, (pixel_x, pixel_y), 2, (0, 255, 0) if pp == 0 else (255, 0, 0), -1)


        # Display the image using OpenCV
        cv2.imshow("Camera Image", rgb_image)
        key = cv2.waitKey(0)  # Wait for a short period to display the image

        
        # Quit if "q" is pressed
        if key == ord('q'):
            break

cv2.destroyAllWindows()  # Close all OpenCV windows