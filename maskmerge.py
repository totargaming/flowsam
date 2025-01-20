import os
import glob
import numpy as np
import cv2

def overlay_mask_on_image(rgb_image, mask_image, top_left, bottom_right):
    # Ensure the mask is binary
    _, binary_mask = cv2.threshold(mask_image, 1, 255, cv2.THRESH_BINARY)

    # Create a region mask
    region_mask = np.zeros_like(binary_mask)
    region_mask[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]] = 255

    # Combine the region mask with the binary mask
    combined_mask = cv2.bitwise_and(binary_mask, region_mask)

    # Create a color mask with green color
    color_mask = np.zeros_like(rgb_image)
    color_mask[combined_mask == 255] = [0, 255, 0]  # Green color for the mask

    # Blend the RGB image and the color mask with low opacity
    blended_image = cv2.addWeighted(rgb_image, 0.7, color_mask, 0.3, 0)

    # Apply the combined mask to keep the original background
    final_image = np.where(combined_mask[..., None] == 255, blended_image, rgb_image)

    return final_image

def process_frames(rgb_folder, mask_folder, output_video_path, top_left, bottom_right, fps=30):
    # Get list of RGB and mask image paths
    rgb_image_paths = sorted(glob.glob(os.path.join(rgb_folder, '*.jpg')))
    mask_image_paths = sorted(glob.glob(os.path.join(mask_folder, '*.png')))

    # Check if the number of RGB images matches the number of mask images
    if len(rgb_image_paths) != len(mask_image_paths):
        raise ValueError("The number of RGB images and mask images do not match.")

    # Read the first frame to get the frame size
    first_frame = cv2.imread(rgb_image_paths[0])
    height, width, _ = first_frame.shape

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    for rgb_image_path in rgb_image_paths:
        # Get the corresponding mask image path
        mask_image_path = os.path.join(mask_folder, os.path.basename(rgb_image_path).replace('.jpg', '.png'))

        # Read the RGB image
        rgb_image = cv2.imread(rgb_image_path)
        if rgb_image is None:
            raise FileNotFoundError(f"RGB image file not found: {rgb_image_path}")

        # Read the mask image
        mask_image = cv2.imread(mask_image_path, cv2.IMREAD_GRAYSCALE)
        if mask_image is None:
            raise FileNotFoundError(f"Mask image file not found: {mask_image_path}")

        # Overlay the mask on the RGB image
        blended_image = overlay_mask_on_image(rgb_image, mask_image, top_left, bottom_right)

        # Write the frame to the video
        video_writer.write(blended_image)

    # Release the video writer
    video_writer.release()

# Example usage
rgb_folder = 'framesList/JPEGImages/hand'
mask_folder = 'nonhung/hand'
output_video_path = 'output_video.mp4'
top_left = (0,800)  # Replace with actual top-left coordinates
bottom_right = (1400, 1400)  # Replace with actual bottom-right coordinates

process_frames(rgb_folder, mask_folder, output_video_path, top_left, bottom_right, fps=30)