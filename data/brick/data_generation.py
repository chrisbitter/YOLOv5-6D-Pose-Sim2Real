import pybullet as p
import os
import numpy as np
import cv2
import trimesh
import os
import logging
import argparse
import random
import shutil

parser = argparse.ArgumentParser()
parser.add_argument('--brick_name', type=str, default="brick4x2", help='Name of the brick (default: brick4x2)')
parser.add_argument('--outdir', type=str, default=None, help='Output directory (default: None, will be created in the brick directory)')
parser.add_argument('--total_samples', type=int, default=100, help='Total number of samples to generate (default: 100)')
parser.add_argument('--overwrite', action='store_true', help='Overwrite existing data')
parser.add_argument('--append', action='store_true', help='Append to existing data')
parser.add_argument('--width', type=int, default=1280, help='Width of the image (default: 1280)')
parser.add_argument('--height', type=int, default=720, help='Height of the image (default: 720)')
parser.add_argument('--class_label', type=int, default=0, help='Class label (default: 0)')
parser.add_argument('--debug', action='store_true', help='Mode debug')

args = parser.parse_args()
brick_name = args.brick_name
total_samples = args.total_samples
overwrite = args.overwrite
append = args.append
outdir = args.outdir

if outdir is None:
    outdir = brick_name

labels_dir = os.path.join(os.path.dirname(__file__), outdir, "labels")
img_dir = os.path.join(os.path.dirname(__file__), outdir, "JPEGImages")
mask_dir = os.path.join(os.path.dirname(__file__), outdir, "mask")
w, h = args.width, args.height
class_label = args.class_label
debug = args.debug

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG if debug else logging.INFO)

# Connect to PyBullet
p.connect(p.GUI if debug else p.DIRECT)

assert not (overwrite and append), "Cannot overwrite and append at the same time"

if not overwrite and not append:
    list_of_images = os.listdir(img_dir)
    if len(list_of_images) > 0:
        logger.info("Data already exists, skipping generation")
        exit()
elif append:
    logger.info("Appending to existing data")
else:
    logger.info("Overwriting data")
    for folder in [labels_dir, img_dir, mask_dir]:
        try:
            shutil.rmtree(folder)
        except:
            pass

os.makedirs(labels_dir, exist_ok=True)
os.makedirs(img_dir, exist_ok=True)
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
focal_x = 949.8506622
focal_y = 949.70506462
x_offset = 627.84604663
y_offset = 354.06980147

projection_matrix = [1.484141659687981, 0.0, 0.0, 0.0, 0.0, 2.638069623943646, 0.0, 0.0, 0.018990552138761174, -0.016472773698662655, -1.02020202020202, -1.0, 0.0, 0.0, -0.020202020202020204, 0.0]

projection_matrix_tf = np.array(projection_matrix).reshape(4, 4)

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
    
    camera_up = np.random.uniform(low=[-1, -1, 1], high=[1, 1, 1])
    camera_up /= np.linalg.norm(camera_up)  # Normalize the camera_up vector
    # camera_up = [-1, 0, 0]      # Adjust the up direction to maintain a proper view

    view_matrix = p.computeViewMatrix(camera_position, camera_target, camera_up)
    view_matrix_tf = np.array(view_matrix).reshape(4, 4).T

    brick_color = np.random.rand(3)
    brick_color /= np.linalg.norm(brick_color)
    brick_color *= np.random.uniform(0.5, 1)
    brick_color = brick_color.tolist()

    brick_color += [1]
    p.changeVisualShape(brick_id, -1, rgbaColor=brick_color)  # Update the brick's color

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

    coordinates[:, 1] = -coordinates[:, 1]
    coordinates += 1
    coordinates /= 2

    x_range = coordinates[:, 0].max() - coordinates[:, 0].min()
    y_range = coordinates[:, 1].max() - coordinates[:, 1].min()


    label = f'{class_label} {" ".join(f"{value:.6f}" for value in coordinates.flatten().tolist())} {x_range:.6f} {y_range:.6f} {focal_x:.6f} {focal_y:.6f} {width} {height} {x_offset:.6f} {y_offset:.6f} {width} {height}'

    file_path = os.path.join(labels_dir, f"{sample_id:05}{class_label}.txt")

    with open(file_path, "w") as f:
        f.write(label)

    img_path = os.path.join(img_dir, f"{sample_id:05}{class_label}.jpg")
    rgb_image = np.array(rgb)
    rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR) 
    cv2.imwrite(img_path, rgb_image)

    mask_path = os.path.join(mask_dir, f"{sample_id:03}{class_label}.png")
    seg += 1
    seg *= 255
    cv2.imwrite(mask_path, seg)

    sample_id += 1
    progress_bar.update(1)
    if debug:
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

try:
    cv2.destroyAllWindows()  # Close all OpenCV windows
except:
    pass

list_of_images = os.listdir(img_dir)
shuffled_list = random.shuffle(list_of_images)

print(len(list_of_images))

train_txt = os.path.join(os.path.dirname(__file__), outdir, "train.txt")
train_range_txt = os.path.join(os.path.dirname(__file__), outdir, "training_range.txt")
test_txt = os.path.join(os.path.dirname(__file__), outdir, "test.txt")
validation_txt = os.path.join(os.path.dirname(__file__), outdir, "validation.txt")

train = list_of_images[:int(len(list_of_images)*0.8)]
test = list_of_images[int(len(list_of_images)*0.8):int(len(list_of_images)*0.9)]
validation = list_of_images[int(len(list_of_images)*0.9):]

print(len(train), len(test), len(validation))

# write the lists to a file
with open(train_txt, "w") as f:
    for item in train[:-1]:
        f.write("%s\n" % item)
    
    f.write(train[-1])

with open(train_range_txt, "w") as f:
    for item in train[:-1]:
        sample_id, _ = item.split(".")
        f.write("%s\n" % int(sample_id))

    sample_id, _ = train[-1].split(".")
    f.write("%s\n" % int(sample_id))

with open(test_txt, "w") as f:
    for item in test[:-1]:
        f.write("%s\n" % item)

    f.write(test[-1])

with open(validation_txt, "w") as f:
    for item in validation[:-1]:
        f.write("%s\n" % item)

    f.write(validation[-1])
