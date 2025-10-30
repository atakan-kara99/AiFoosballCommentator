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


# Callback function for mouse events
def draw_circle(event, x, y, flags, param):
    global ix, iy, drawing, image_rgb, mask, radius
    p1 = (x - radius, y - radius)
    p2 = (x + radius, y + radius)
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
        # Draw a circle immediately on mouse down
        cv2.rectangle(image_rgb, p1, p2, (0, 0, 255), -1)  # Red circle on RGB image
        cv2.rectangle(mask, p1, p2, 255, -1)  # White circle on mask
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            # Draw a circle while moving the mouse
            cv2.rectangle(
                image_rgb,
                p1,
                p2,
                (0, 0, 255),
                -1,
            )
            cv2.rectangle(mask, p1, p2, 255, -1)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        # Draw a circle on mouse release (finalize the drawing)
        cv2.rectangle(image_rgb, p1, p2, (0, 0, 255), -1)
        cv2.rectangle(mask, p1, p2, 255, -1)


# Function to process the images in the directory
def label_images_in_directory(image_dir):
    global image_rgb, mask

    # Get the list of image files in the directory
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith((".png"))]

    # Iterate over each image in the directory
    for image_file in image_files:
        image_path = os.path.join(image_dir, image_file)
        # check if the image already exists in the target directory
        if os.path.isfile(os.path.join(image_dir, f"../masks/{image_file}")):
            print(f"{image_file} already labeled")
            continue

        print(f"Processing {image_file}")

        # Load the image (both in color and grayscale for the mask)
        image_rgb = cv2.imread(image_path)
        training_img = image_rgb.copy()
        mask = np.zeros_like(
            image_rgb[:, :, 0]
        )  # Grayscale mask (same size as the image)

        # Set up the mouse callback to draw
        cv2.namedWindow("Image")
        cv2.setMouseCallback("Image", draw_circle)

        while True:
            # Display the RGB image with drawing
            cv2.imshow("Image", image_rgb)

            # Wait for a key event
            key = cv2.waitKey(1) & 0xFF

            # If 'n' is pressed, save the mask and go to the next image
            if key == ord("n"):
                head, _ = os.path.split(image_dir)
                mask_filename = os.path.join(head, f"masks/{image_file}")
                training_image_filename = os.path.join(head, f"images/{image_file}")

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
image_dir = "cv/player_detection/training_images/unlabeled_images"  # Specify your image directory here
label_images_in_directory(image_dir)
