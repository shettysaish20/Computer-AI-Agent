import pyautogui
import time
import asyncio
from main import process_os_image
import numpy as np # For image conversion
import cv2 # For image conversion
import pygetwindow # For targeting a specific window

# Assume you have a function to run your CV pipeline
async def run_cv_pipeline_on_screenshot(screenshot_image) -> list[dict]:
    detected_elements = await process_os_image(screenshot_image)
    # ... your CV pipeline logic ...
    # Each element in the list should be a dict like:
    # {'text': 'button_label', 'bbox': [xmin, ymin, xmax, ymax]}
    return detected_elements # type: ignore

def find_element_in_cv_output(cv_elements, target_text):
    for element in cv_elements:
        if target_text.strip().lower() in element.get('g_icon_name', '').strip().lower():
            return element
    return None

async def main_task():
    task_description = "Calculate 12 + 34"
    # Sequence of button labels to press for this specific task
    action_sequence = ["1", "2", "+", "3", "4", "="]

    # Ensure Calculator is open and visible (manual step for this basic version)
    print(f"Starting task: {task_description}")
    print("Please ensure the Calculator app is open and visible.")
    time.sleep(3) # Give user time

    for target_label in action_sequence:
        print(f"Attempting to press: '{target_label}'")

        # a. Capture Screenshot
        screenshot = pyautogui.screenshot()
        # screenshot.save("current_screen.png") # Optional: for debugging

        # b. Run CV Pipeline
        # You'll need to adapt this to how your CV pipeline is called
        # For example, if it takes a file path:
        # screenshot.save("temp_screen.png")
        # detected_elements = await run_cv_pipeline_on_image_path("temp_screen.png")
        # Or if it takes an image object (e.g., PIL Image, which pyautogui.screenshot() returns):
        detected_elements = await run_cv_pipeline_on_screenshot(screenshot)

        if not detected_elements:
            print("Error: CV pipeline returned no elements.")
            break

        # c. Find Target Element by Label
        target_element = find_element_in_cv_output(detected_elements, target_label)

        if target_element and 'bbox' in target_element:
            bbox = target_element['bbox'] # Assuming [xmin, ymin, xmax, ymax]
            # Calculate center of the bounding box
            click_x = (bbox[0] + bbox[2]) / 2
            click_y = (bbox[1] + bbox[3]) / 2

            # d. Perform Action
            print(f"  Found '{target_label}' at bbox {bbox}. Clicking at ({click_x:.0f}, {click_y:.0f}).")
            pyautogui.click(click_x, click_y)
            await asyncio.sleep(0.75) # Wait for UI to react and for next screenshot to be accurate
        else:
            print(f"  Error: Could not find element '{target_label}' on screen or it has no bbox.")
            print(f"  Detected elements: {[el.get('text') for el in detected_elements]}") # Debug
            # Consider adding a fallback or error handling here
            break

    print("Task sequence completed (or failed).")

if __name__ == "__main__":
    asyncio.run(main_task())