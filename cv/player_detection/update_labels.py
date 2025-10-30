import cv2
import os
import numpy as np

# Global variables for drawing
drawing = False  # Whether the user is drawing
ix, iy = -1, -1  # Initial position of mouse
radius = 6  # Circle radius for drawing

# Create a mask to store the drawn area
mask = None
image_rgb = None
training_img = None


def apply_mask(image, mask, color, inv=False):
    if inv:
        mask = cv2.bitwise_not(mask)
    bright_red = np.full_like(image, color)
    image[mask > 0] = bright_red[mask > 0]
    return image


# Callback function for mouse events
def draw_circle(event, x, y, flags, param):
    global ix, iy, drawing, image_rgb, mask, radius, training_img
    p1 = (x - radius, y - radius)
    p2 = (x + radius, y + radius)
    if flags == 1:  # primary mouse button -> draw rectangle
        cv2.rectangle(mask, p1, p2, 255, -1)
        image_rgb = apply_mask(training_img.copy(), mask, (0, 0, 255))
    elif flags == 2:  # secondary mouse button -> erase
        cv2.rectangle(mask, p1, p2, 0, -1)
        image_rgb = apply_mask(training_img.copy(), mask, (0, 0, 255))


# Function to process the images in the directory
def label_images_in_directory(image_dir):
    global training_img, image_rgb, mask

    # Get the list of image files in the directory
    image_files = [
        f
        for f in os.listdir(os.path.join(image_dir, "images"))
        if f.lower().endswith((".png"))
    ]
    print(f"image files: {len(image_files)}")
    print(image_files)
    # Iterate over each image in the directory
    for image_file in image_files:
        image_path = os.path.join(image_dir, "images", image_file)
        mask_path = os.path.join(image_dir, "masks", image_file)
        # check if the image already exists in the target directory
        if os.path.isfile(os.path.join(image_dir, f"../masks/{image_file}")):
            print(f"{image_file} already labeled")
            continue

        print(f"Processing {image_file}")

        # Load the image (both in color and grayscale for the mask)
        image_rgb = cv2.imread(image_path)
        training_img = image_rgb.copy()
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        apply_mask(image_rgb, mask, (0, 0, 255))

        # Set up the mouse callback to draw
        cv2.namedWindow("Image")
        cv2.setMouseCallback("Image", draw_circle)

        while True:
            # Display the RGB image with drawing
            cv2.imshow("Image", image_rgb)
            cv2.imshow("mask", mask)

            # Wait for a key event
            key = cv2.waitKey(1) & 0xFF

            # If 'n' is pressed, save the mask and go to the next image
            if key == ord("n"):

                mask_filename = os.path.join(image_dir, "masks", image_file)
                training_image_filename = os.path.join(image_dir, "images", image_file)

                cv2.imwrite(mask_filename, mask)  # Save the mask image
                cv2.imwrite(
                    training_image_filename, training_img
                )  # Save the mask image
                print(f"Mask saved as {mask_filename}")
                print(f"Training image saved as {training_image_filename}")
                break

            # skip this image
            if key == ord("s"):
                print("skipped this image...")
                break

            # If 'q' is pressed, exit the loop
            if key == ord("q"):
                print("Exiting...")
                return

        # Close the current image window
        cv2.destroyAllWindows()


# Example usage:
image_dir = (
    "cv/player_detection/training_images/test_003"  # Specify your image directory here
)
image_dir = os.path.normpath(image_dir)
label_images_in_directory(image_dir)
