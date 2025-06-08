import pyautogui
import time
import asyncio
from main import process_os_image
import numpy as np
import cv2
import pygetwindow
import os
import subprocess # For launching calculator

# Define a temporary image path
TEMP_IMAGE_DIR = "temp_screenshots"
os.makedirs(TEMP_IMAGE_DIR, exist_ok=True)
TEMP_IMAGE_PATH = os.path.join(TEMP_IMAGE_DIR, "initial_calculator_state.png")

async def run_cv_pipeline_on_image_path(image_path: str) -> list[dict]:
    detected_elements = await process_os_image(image_path=image_path)
    return detected_elements if detected_elements else []

def find_element_in_cv_output(cv_elements, target_text):
    if not cv_elements: # Add a check for empty or None cv_elements
        return None
    for element in cv_elements:
        # Using 'in' for partial match as Gemini names can be descriptive
        if target_text.strip().lower() == element.get('g_icon_name', '').strip().lower():
            return element
    return None

async def launch_and_prepare_calculator():
    """Tries to launch, find, activate, and maximize the Calculator."""
    calculator_window = None
    try:
        print("Attempting to launch Calculator...")
        subprocess.Popen("calc.exe")
        time.sleep(2) # Give Calculator time to open

        possible_titles = ["Calculator", "Calculatrice", "Rechner", "Calcolatrice", "Calculadora"]
        for _ in range(5): # Try a few times to find the window
            for title in possible_titles:
                windows = pygetwindow.getWindowsWithTitle(title)
                if windows:
                    calculator_window = windows[0]
                    break
            if calculator_window:
                break
            time.sleep(0.5)
        
        if calculator_window:
            print(f"Found Calculator window: '{calculator_window.title}'")
            if calculator_window.isMinimized:
                calculator_window.restore()
                time.sleep(0.2)
            calculator_window.activate()
            time.sleep(0.2)
            # Maximize is generally more reliable than true fullscreen for automation
            if not calculator_window.isMaximized:
                calculator_window.maximize()
            time.sleep(0.5) # Allow time for maximize animation
            
            if calculator_window.isMaximized:
                print("  Calculator window maximized.")
                return calculator_window
            else:
                print("  Warning: Could not confirm Calculator window is maximized.")
                return calculator_window # Return even if not confirmed maximized
        else:
            print("Error: Calculator window not found after launch.")
            return None
    except Exception as e:
        print(f"Error during Calculator launch/preparation: {e}")
        return None

async def main_task():
    task_description = "Calculate 12 + 34"
    action_sequence = ["1", "2", "+", "3", "4", "="]
    # For calculator, "equals" is often the label for the = button
    # Or "display" for the results area if you were reading it.
    # Let's assume your CV pipeline labels the '=' button as "equals" or contains "equals"

    print(f"Starting task: {task_description}")
    
    all_detected_elements = None
    cv_pipeline_run_successfully = False
    calculator_window = await launch_and_prepare_calculator()

    if calculator_window:
        print("Calculator prepared. Taking initial screenshot for CV pipeline...")
        pil_image = None
        capture_successful = False
        try:
            # Ensure window has valid dimensions before screenshot
            if calculator_window.width > 0 and calculator_window.height > 0:
                pil_image = pyautogui.screenshot(region=(
                    calculator_window.left,
                    calculator_window.top,
                    calculator_window.width,
                    calculator_window.height
                ))
                print(f"  Captured region: {calculator_window.left}, {calculator_window.top}, {calculator_window.width}, {calculator_window.height}")
                capture_successful = True
            else:
                print("  Calculator window has invalid dimensions. Falling back to full screen.")
                pil_image = pyautogui.screenshot()
                capture_successful = True # Still attempt full screen
                calculator_window = None # Invalidate window specific offsets if full screen

        except Exception as e:
            print(f"  Error capturing specific window region: {e}. Falling back to full screen.")
            pil_image = pyautogui.screenshot() # Fallback
            capture_successful = True
            calculator_window = None # Invalidate window specific offsets

        if not capture_successful or pil_image is None:
            print("Error: Failed to capture screenshot for CV pipeline.")
        else:
            try:
                frame_rgb = np.array(pil_image)
                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                cv2.imwrite(TEMP_IMAGE_PATH, frame_bgr)
                print(f"  Initial screenshot saved to: {TEMP_IMAGE_PATH}")

                all_detected_elements = await run_cv_pipeline_on_image_path(TEMP_IMAGE_PATH)
                if all_detected_elements:
                    print(f"  CV Pipeline successful. Found {len(all_detected_elements)} elements.")
                    cv_pipeline_run_successfully = True
                else:
                    print(f"  Error: CV pipeline returned no elements for image: {TEMP_IMAGE_PATH}")
            except Exception as e:
                print(f"  Error during initial CV processing or saving screenshot: {e}")
    else:
        print("Calculator not prepared. Please ensure it's open, maximized, and visible.")
        print("You might need to run the script with administrator privileges for window control.")
        # Optionally, you could fall back to asking the user to prepare and then taking a full screenshot.
        # For now, we'll assume if auto-prep fails, we can't proceed reliably with single CV run.
        return # Exit if calculator setup failed

    if not cv_pipeline_run_successfully:
        print("Cannot proceed with interactions as initial CV analysis failed.")
        return

    # --- Interaction Loop using the single set of detected elements ---
    for target_label in action_sequence:
        print(f"Attempting to press: '{target_label}' using stored CV data.")

        target_element = find_element_in_cv_output(all_detected_elements, target_label)

        if target_element and 'bbox' in target_element:
            bbox = target_element['bbox'] 
            
            offset_x, offset_y = 0, 0
            # If calculator_window is valid, it means we likely captured that region.
            # The bbox from CV is relative to that captured region.
            # pyautogui.click needs global screen coordinates.
            if calculator_window and calculator_window.width > 0 and calculator_window.height > 0:
                offset_x = calculator_window.left
                offset_y = calculator_window.top
            
            center_x_in_image = (bbox[0] + bbox[2]) / 2
            center_y_in_image = (bbox[1] + bbox[3]) / 2
            
            click_x = offset_x + center_x_in_image
            click_y = offset_y + center_y_in_image

            print(f"  Found '{target_label}' (e.g., {target_element.get('g_icon_name')}) at CV bbox {bbox}. Clicking at global screen coords ({click_x:.0f}, {click_y:.0f}).")
            pyautogui.click(click_x, click_y)
            await asyncio.sleep(0.75) # Wait for UI to react
        else:
            print(f"  Error: Could not find element '{target_label}' in stored CV data.")
            print(f"  Available elements were: {[el.get('g_icon_name', 'N/A') for el in all_detected_elements if el]}")
            break # Stop if an element can't be found
    
    print("Task sequence completed (or failed).")

if __name__ == "__main__":
    asyncio.run(main_task())