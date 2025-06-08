import pyautogui
import time
import asyncio
from main import process_os_image
import numpy as np # For image conversion
import cv2 # For image conversion
import pygetwindow # For targeting a specific window
import os # For creating a temporary file path

# Define a temporary image path
TEMP_IMAGE_DIR = "temp_screenshots"
os.makedirs(TEMP_IMAGE_DIR, exist_ok=True) # Ensure the directory exists
TEMP_IMAGE_PATH = os.path.join(TEMP_IMAGE_DIR, "temp_screenshot.png")


# Assume you have a function to run your CV pipeline
async def run_cv_pipeline_on_image_path(image_path: str) -> list[dict]:
    detected_elements = await process_os_image(image_path=image_path)
    return detected_elements # type: ignore

def find_element_in_cv_output(cv_elements, target_text):
    for element in cv_elements:
        # Assuming your CV output elements have 'g_icon_name' after Gemini processing
        if target_text.strip().lower() == element.get('g_icon_name', '').strip().lower():
            return element
    return None

async def main_task():
    task_description = "Calculate 12 + 34"
    action_sequence = ["1", "2", "+", "3", "4", "="]

    print(f"Starting task: {task_description}")
    print("Attempting to focus and capture the Calculator app.")
    
    calculator_window = None
    try:
        possible_titles = ["Calculator", "Calculatrice", "Rechner", "Calcolatrice", "Calculadora"]
        for title in possible_titles:
            windows = pygetwindow.getWindowsWithTitle(title)
            if windows:
                calculator_window = windows[0]
                break
        
        if calculator_window:
            print(f"Found Calculator window: '{calculator_window.title}'")
            if calculator_window.isMinimized:
                calculator_window.restore()
            calculator_window.activate() 
            time.sleep(0.5) 
        else:
            print("Warning: Calculator window not found by title. Will capture active screen.")
            print("Please ensure the Calculator app is open and visible.")
            time.sleep(3) 

    except Exception as e:
        print(f"Error finding/activating Calculator window: {e}. Will capture active screen.")
        print("Please ensure the Calculator app is open and visible.")
        time.sleep(3)

    for target_label in action_sequence:
        print(f"Attempting to press: '{target_label}'")

        # a. Capture Screenshot
        pil_image = None
        capture_successful = False
        if calculator_window and calculator_window.width > 0 and calculator_window.height > 0 :
            try:
                pil_image = pyautogui.screenshot(region=(
                    calculator_window.left,
                    calculator_window.top,
                    calculator_window.width,
                    calculator_window.height
                ))
                print(f"  Captured region: {calculator_window.left}, {calculator_window.top}, {calculator_window.width}, {calculator_window.height}")
                capture_successful = True
            except Exception as e:
                print(f"  Error capturing calculator window region: {e}. Falling back to full screen.")
                pil_image = pyautogui.screenshot()
                capture_successful = True # Still captured something
        else:
            pil_image = pyautogui.screenshot()
            print("  Captured full screen.")
            capture_successful = True

        if not capture_successful or pil_image is None:
            print("Error: Failed to capture screenshot.")
            break
        
        # b. Convert PIL Image to OpenCV BGR format and Save
        try:
            frame_rgb = np.array(pil_image)
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(TEMP_IMAGE_PATH, frame_bgr)
            print(f"  Screenshot saved to: {TEMP_IMAGE_PATH}")
        except Exception as e:
            print(f"Error saving screenshot: {e}")
            break

        # c. Run CV Pipeline using the image path
        detected_elements = await run_cv_pipeline_on_image_path(TEMP_IMAGE_PATH)

        if not detected_elements:
            print(f"Error: CV pipeline returned no elements for image: {TEMP_IMAGE_PATH}")
            # Optionally, keep the image that resulted in no detections by not deleting it
            # or copying it elsewhere.
            break

        # d. Find Target Element by Label
        target_element = find_element_in_cv_output(detected_elements, target_label)

        if target_element and 'bbox' in target_element:
            bbox = target_element['bbox'] 
            
            offset_x, offset_y = 0, 0
            if calculator_window and calculator_window.width > 0 and calculator_window.height > 0:
                offset_x = calculator_window.left
                offset_y = calculator_window.top
            
            center_x_in_image = (bbox[0] + bbox[2]) / 2
            center_y_in_image = (bbox[1] + bbox[3]) / 2
            
            click_x = offset_x + center_x_in_image
            click_y = offset_y + center_y_in_image

            # e. Perform Action
            print(f"  Found '{target_label}' at CV bbox {bbox}. Clicking at global screen coords ({click_x:.0f}, {click_y:.0f}).")
            pyautogui.click(click_x, click_y)
            await asyncio.sleep(0.75) 
        else:
            print(f"  Error: Could not find element '{target_label}' on screen or it has no bbox.")
            print(f"  Available elements from CV: {[el.get('g_icon_name', 'N/A') for el in detected_elements]}")
            break
    
    # Optional: Clean up the temporary image file after the task sequence
    # try:
    #     if os.path.exists(TEMP_IMAGE_PATH):
    #         os.remove(TEMP_IMAGE_PATH)
    #         print(f"Cleaned up temporary image: {TEMP_IMAGE_PATH}")
    # except Exception as e:
    #     print(f"Error cleaning up temporary image: {e}")

    print("Task sequence completed (or failed).")

if __name__ == "__main__":
    asyncio.run(main_task())