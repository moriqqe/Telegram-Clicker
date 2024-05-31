import cv2
import pyautogui
import numpy as np
import time
from mss import mss

# Load the target image
target_image_path = 'blum.png'
target = cv2.imread(target_image_path, cv2.IMREAD_GRAYSCALE)


# Function to locate the target icon on the screen
def locate_target_on_screen(screen_gray, target_gray, threshold=0.8):
    # Define the scales to search at
    scales = np.linspace(0.5, 1.5, 5)[::-1]  # Reduce the number of scales for speed

    for scale in scales:
        # Resize the target image
        resized_target = cv2.resize(target_gray, (int(target_gray.shape[1] * scale), int(target_gray.shape[0] * scale)))

        # Perform template matching to find the target image
        result = cv2.matchTemplate(screen_gray, resized_target, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

        if max_val >= threshold:
            # Calculate the center of the detected icon
            target_height, target_width = resized_target.shape
            center_x = max_loc[0] + target_width // 2
            center_y = max_loc[1] + target_height // 2
            return center_x, center_y, scale
    return None, None, None


# Function to perform a mouse click at the specified location
def click_at_location(x, y):
    pyautogui.click(x, y)


# Initialize click counter
click_counter = 0

# Main loop to continuously search and click on the target icon
start_time = time.time()
sct = mss()

# Define the screen region to capture (full screen)
monitor = sct.monitors[1]

while True:
    # Capture the screen
    screen = sct.grab(monitor)
    screen_np = np.array(screen)
    screen_gray = cv2.cvtColor(screen_np, cv2.COLOR_BGR2GRAY)

    x, y, scale = locate_target_on_screen(screen_gray, target)
    if x is not None and y is not None:
        click_counter += 1
        click_at_location(x, y)
        print(f"Click #{click_counter} at ({x}, {y}), scale: {scale:.2f}")

    # Ensure the loop runs at the desired click rate (50 clicks per second)
    elapsed_time = time.time() - start_time
    if elapsed_time < 1:
        if click_counter >= 50:
            time.sleep(1 - elapsed_time)
            start_time = time.time()
            click_counter = 0
