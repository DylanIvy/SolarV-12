import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

def compute_cloud_coverage(frame):
    """
    Estimates cloud coverage percentage from a given HSV video frame and returns the cloud mask.

    Args:
        frame (np.ndarray): HSV image frame (shape: H, W, 3)

    Returns:
        float: Estimated cloud coverage percentage (0-100)
        np.ndarray: Cloud mask (binary image)
    """





    # Extract the Hue, Saturation, and Value channels
    hue_channel = frame[:, :, 0]
    saturation_channel = frame[:, :, 1]
    value_channel = frame[:, :, 2]

    # Define thresholds for clouds
    # Hue range for pink/purple clouds
    lower_pink = 140
    upper_pink = 160

    # Saturation range for grey/dull clouds (low saturation)
    lower_saturation = 0
    upper_saturation = 50  # Adjust this based on your images

    # Value range for bright clouds
    lower_value = 100  # Adjust this based on your images
    upper_value = 255

    # Create masks for each condition
    # Mask for pink/purple clouds (based on hue)
    hue_mask = cv2.inRange(hue_channel, lower_pink, upper_pink)

    # Mask for grey/dull clouds (based on low saturation and high brightness)
    saturation_mask = cv2.inRange(saturation_channel, lower_saturation, upper_saturation)
    value_mask = cv2.inRange(value_channel, lower_value, upper_value)
    grey_cloud_mask = cv2.bitwise_and(saturation_mask, value_mask)

    # Combine the masks to detect all types of clouds
    cloud_mask = cv2.bitwise_or(hue_mask, grey_cloud_mask)

    # Calculate cloud coverage as a percentage
    total_pixels = cloud_mask.size
    cloud_pixels = np.count_nonzero(cloud_mask)
    cloud_coverage = (cloud_pixels / total_pixels) * 100

    return cloud_coverage
def extract_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    cloud_coverage_list = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # OpenCV loads images as BGR; convert to HSV:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        cloud_coverage= compute_cloud_coverage(frame)
        cloud_coverage_list.append(cloud_coverage)

        frames.append(frame)
    cap.release()
    return np.array(frames), np.array(cloud_coverage_list)

# List of video files, have to run twice for full training video and full verification video.
video_files = [
    r"C:\Users\jldag\OneDrive\Desktop\Senior-Design\sky_images\1-1-24.mp4",
    r"C:\Users\jldag\OneDrive\Desktop\Senior-Design\sky_images\1-2-24.mp4",
    r"C:\Users\jldag\OneDrive\Desktop\Senior-Design\sky_images\1-3-24.mp4",
    r"C:\Users\jldag\OneDrive\Desktop\Senior-Design\sky_images\1-4-24.mp4"
]

# Merge videos (same as before)
output_folder = r"C:\Users\jldag\OneDrive\Desktop\Senior-Design\sky_images"
os.makedirs(output_folder, exist_ok=True)
output_file = os.path.join(output_folder, "output.mp4")
fps = 30

# Get properties of the first video
cap = cv2.VideoCapture(video_files[0])
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
cap.release()

out = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

for video in video_files:
    cap = cv2.VideoCapture(video)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
    cap.release()

out.release()
print("Videos merged successfully!")

# Extract frames and cloud masks
video_path = r"C:\Users\jldag\OneDrive\Desktop\Senior-Design\sky_images\1-10-24.mp4"
video_frames, cloud_coverage_data= extract_frames(video_path)
print(f"Extracted {len(video_frames)} frames from the video.")
print(f"Cloud Coverage data length {len(cloud_coverage_data)}")

# Display the cloud mask for a specific frame
#frame_index = 400 #Choose the frame index to display
#plt.figure(figsize=(10, 5))

# Display the original frame
#plt.subplot(1, 2, 1)
#plt.imshow(cv2.cvtColor(video_frames[frame_index], cv2.COLOR_HSV2RGB))
#plt.title("Original Frame")
#plt.axis("off")



# Display the cloud mask
#plt.subplot(1, 2, 2)
#plt.imshow(cloud_masks[frame_index], cmap="gray")
#plt.title("Cloud Mask")
#plt.axis("off")

#plt.show()