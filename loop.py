import pyautogui
import time
import asyncio
from main import process_os_image
import numpy as np
import cv2
import pygetwindow
import os
import subprocess # For launching calculator
import json # For parsing Gemini's output

# For Gemini
from google import genai
from dotenv import load_dotenv

# Load environment variables (for GEMINI_API_KEY)
load_dotenv()

# Configure Gemini API
try:
    api_key = os.getenv("GEMINI_API_KEY") # Changed from GEMINI_API_KEY to GOOGLE_API_KEY as per common practice
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in environment variables.")
    # Using the client approach as in your latest working version
    gemini_client = genai.Client(api_key=api_key)
    print("Gemini client initialized successfully.")
except Exception as e:
    print(f"Error configuring Gemini API: {e}")


# Define a temporary image path
TEMP_IMAGE_DIR = "temp_screenshots"
os.makedirs(TEMP_IMAGE_DIR, exist_ok=True)
TEMP_IMAGE_PATH = os.path.join(TEMP_IMAGE_DIR, "initial_calculator_state.png")

async def run_cv_pipeline_on_image_path(image_path: str) -> list[dict]:
    detected_elements = await process_os_image(image_path=image_path)
    return detected_elements if detected_elements else []

def find_element_in_cv_output(cv_elements, target_text):
    if not cv_elements: 
        return None
    for element in cv_elements:
        if target_text.strip().lower() == element.get('g_icon_name', '').strip().lower():
            return element
    return None

async def launch_and_prepare_calculator():
    """Tries to launch, find, activate, and maximize the Calculator.
    Returns the window object and its geometry if successful."""
    calculator_window = None
    try:
        print("Attempting to launch Calculator...")
        subprocess.Popen("calc.exe")
        time.sleep(2) 

        possible_titles = ["Calculator", "Calculatrice", "Rechner", "Calcolatrice", "Calculadora"]
        for _ in range(5): 
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
            
            # Attempt activation multiple times if needed, with small delays
            activated = False
            for attempt in range(3):
                try:
                    calculator_window.activate()
                    time.sleep(0.3) # Increased delay after activate
                    if calculator_window.isActive:
                        activated = True
                        break
                except Exception as act_e:
                    print(f"  Activation attempt {attempt+1} failed: {act_e}")
                print(f"  Retrying activation (attempt {attempt+2})...")
                time.sleep(0.5)
            
            if not activated:
                print("  Warning: Could not confirm Calculator window is active after multiple attempts.")
                # Optionally, try a click to focus as a last resort
                try:
                    pyautogui.click(calculator_window.centerx, calculator_window.centery)
                    time.sleep(0.2)
                    print("  Clicked center of window as a focus fallback.")
                except Exception:
                    pass # Ignore if this fails

            if not calculator_window.isMaximized:
                calculator_window.maximize()
            time.sleep(0.5) 
            
            # Crucially, re-fetch the window attributes *after* all operations to get final state
            # This helps if maximize() or activate() changed them.
            # However, for screenshot stability, we'll capture them *before* screenshotting.
            # The return values will be what we use for consistent offsets.

            if calculator_window.isMaximized: # Check again after maximize
                print("  Calculator window maximized.")
                # Return a snapshot of its geometry at this point
                return calculator_window, {
                    "left": calculator_window.left, 
                    "top": calculator_window.top, 
                    "width": calculator_window.width, 
                    "height": calculator_window.height
                }
            else:
                print("  Warning: Could not confirm Calculator window is maximized.")
                # Still return it and its current geometry
                return calculator_window, {
                    "left": calculator_window.left, 
                    "top": calculator_window.top, 
                    "width": calculator_window.width, 
                    "height": calculator_window.height
                }
        else:
            print("Error: Calculator window not found after launch.")
            return None, None
    except Exception as e:
        print(f"Error during Calculator launch/preparation: {e}")
        return None, None

def clean_gemini_response(text_response: str) -> str:
    """Cleans Gemini's response by removing markdown JSON backticks and stripping whitespace."""
    cleaned = text_response.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]
    
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    return cleaned.strip()

async def get_action_sequence_from_gemini(user_task_description: str, available_buttons: list[str]) -> list[str] | None:
    """
    Uses Gemini to generate an action sequence based on user input and available buttons.
    """
    # if not gemini_model:
    #     print("Error: Gemini model not initialized. Cannot generate action sequence.")
    #     return None
    system_prompt = f"""You are an AI assistant that translates natural language tasks into a sequence of button presses for a standard calculator application.
        
        Available buttons:
        {json.dumps(available_buttons)}

        User Task: 
        {user_task_description}
        
        Instructions:
        - The user will provide a calculation task.
        - You will also be provided with a list of available button labels currently visible on the calculator.
        - Your task is to generate a JSON formatted list of strings, where each string is an exact button label from the provided list of available buttons. This list of strings represents the sequence of buttons to press in the correct order to achieve the user's task.
        - Always ensure calculations that expect an explicit result end with the '=' button, if '=' is available in the list of buttons.
        - Only use button labels that are strictly present in the provided 'Available buttons' list. Do not invent or modify button labels.
        - For multiply operations, use 'multiply' or 'x' (lowercase X) as appropriate based on the available buttons. There might be X (captial) which denotes backspace and not multiply.


        Expected JSON Output (example for "calculate 5 plus 3" if 5, +, 3, = are available):
        ```json
        {{
            "action_sequence": ["5", "+", "3", "="]
        }}
        ```

        Now, generate the JSON output for the provided User Task and Available buttons.
        """
    
    print("\n🤖 Asking Gemini to generate action sequence...")
    print(f"   User Task: {user_task_description}")
    # print(f"   Available Buttons: {available_buttons}") # Can be verbose

    try:
        response = await gemini_client.aio.models.generate_content(
            model="gemini-2.0-flash-lite",
            contents=system_prompt
            )
        
        # if not response.parts:
        #     print("Error: Gemini returned an empty response.")
        #     return None
            
        action_sequence_str = response.text
        print(f"   Gemini Raw Response: {action_sequence_str[:200]}...") # type: ignore

        cleaned_response_str = clean_gemini_response(action_sequence_str) # type: ignore

        try:
            # Assuming Gemini directly returns the list if prompted for JSON list
            # If it wraps it in {"action_sequence": [...]}, then .get("action_sequence") is needed
            parsed_json = json.loads(cleaned_response_str)
            if isinstance(parsed_json, dict) and "action_sequence" in parsed_json:
                action_sequence = parsed_json.get("action_sequence", [])
            elif isinstance(parsed_json, list): # If Gemini returns the list directly
                action_sequence = parsed_json
            else:
                print(f"Error: Gemini response JSON is not in expected format (dict with 'action_sequence' or direct list).")
                print(f"Parsed JSON: {parsed_json}")
                return None

        except json.JSONDecodeError as e:
            print(f"Error: Gemini response was not valid JSON. Error: {e}")
            print(f"Cleaned response attempt: {cleaned_response_str}")
            return None

        if not isinstance(action_sequence, list) or not all(isinstance(item, str) for item in action_sequence):
            print("Error: Gemini action_sequence was not a list of strings.")
            print(f"Parsed response: {action_sequence}")
            return None

        for button in action_sequence:
            if button not in available_buttons:
                print(f"Error: Gemini hallucinated a button '{button}' which is not in the available buttons list.")
                print(f"   Generated sequence: {action_sequence}")
                print(f"   Available buttons: {available_buttons}")
                return None
        
        print(f"   ✅ Gemini generated action sequence: {action_sequence}")
        return action_sequence

    except Exception as e:
        print(f"Error during Gemini API call: {e}")
        return None


async def main_task():
    user_task_description = input("Please enter the calculation task (e.g., 'calculate 50 times 3 plus 10'): ")
    if not user_task_description:
        print("No task entered. Exiting.")
        return

    print(f"\nStarting task based on user input: {user_task_description}")
    
    all_detected_elements = []
    cv_pipeline_run_successfully = False
    # calculator_window = None # Will be set by launch_and_prepare_calculator
    initial_window_geometry = None # To store the stable geometry
    action_sequence = None

    # calculator_window_obj is the pygetwindow object, initial_window_geometry is a dict
    calculator_window_obj, initial_window_geometry = await launch_and_prepare_calculator()

    if calculator_window_obj and initial_window_geometry:
        print("Calculator prepared. Taking initial screenshot for CV pipeline...")
        pil_image = None
        capture_successful = False
        try:
            # Use the stored initial_window_geometry for the screenshot region
            if initial_window_geometry["width"] > 0 and initial_window_geometry["height"] > 0:
                pil_image = pyautogui.screenshot(region=(
                    initial_window_geometry["left"],
                    initial_window_geometry["top"],
                    initial_window_geometry["width"],
                    initial_window_geometry["height"]
                ))
                print(f"  Captured region using stored geometry: {initial_window_geometry}")
                capture_successful = True
            else:
                print("  Calculator window has invalid dimensions (from stored geometry). Falling back to full screen.")
                pil_image = pyautogui.screenshot()
                capture_successful = True 
                initial_window_geometry = None # Invalidate stored geometry if we fall back
        except Exception as e:
            print(f"  Error capturing specific window region: {e}. Falling back to full screen.")
            pil_image = pyautogui.screenshot() 
            capture_successful = True
            initial_window_geometry = None # Invalidate stored geometry

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
        print("Calculator not prepared or geometry not obtained. Please ensure it's open, maximized, and visible.")
        print("You might need to run the script with administrator privileges for window control.")
        return 

    if not cv_pipeline_run_successfully:
        print("Cannot proceed with interactions as initial CV analysis failed.")
        return

    # --- Generate action sequence using Gemini ---
    if all_detected_elements:
        available_buttons = [elem.get('g_icon_name', '').strip() for elem in all_detected_elements if elem.get('g_icon_name')]
        available_buttons = sorted(list(set(filter(None, available_buttons)))) # Unique, sorted, non-empty
        print(f"Available buttons extracted from CV output: {available_buttons}")
        
        if not available_buttons:
            print("Error: No button labels extracted from CV output to provide to Gemini.")
            return
        action_sequence = await get_action_sequence_from_gemini(user_task_description, available_buttons)
    
    if not action_sequence: # If Gemini failed or returned None
        print("Failed to generate a valid action sequence from Gemini. Cannot proceed.")
        return

    print("\n▶️ Starting interaction sequence...")
    for target_label in action_sequence:
        print(f"Attempting to press: '{target_label}' using stored CV data.")
        target_element = find_element_in_cv_output(all_detected_elements, target_label)

        if target_element and 'bbox' in target_element:
            bbox = target_element['bbox'] 
            
            offset_x, offset_y = 0, 0
            # Use the STABLE initial_window_geometry for offsets
            if initial_window_geometry and initial_window_geometry["width"] > 0 and initial_window_geometry["height"] > 0:
                offset_x = initial_window_geometry["left"]
                offset_y = initial_window_geometry["top"]
                print(f"  DEBUG: Using STORED window offsets: left={offset_x}, top={offset_y}")
            else:
                # This case means we fell back to full screen screenshot earlier
                print(f"  DEBUG: Not using stored window offsets (likely full screen capture). Offsets will be 0,0.")


            center_x_in_image = (bbox[0] + bbox[2]) / 2
            center_y_in_image = (bbox[1] + bbox[3]) / 2
            
            click_x = offset_x + center_x_in_image
            click_y = offset_y + center_y_in_image

            print(f"  Found '{target_label}' (exact match: {target_element.get('g_icon_name')}) at CV bbox {bbox}. Clicking at global screen coords ({click_x:.0f}, {click_y:.0f}).")
            
            screen_width, screen_height = pyautogui.size()
            if not (0 <= click_x < screen_width and 0 <= click_y < screen_height):
                print(f"  CRITICAL ERROR: Calculated click coordinates ({click_x:.0f}, {click_y:.0f}) are outside screen bounds ({screen_width}x{screen_height}). Skipping click.")
                print(f"    STORED Offsets: x={offset_x}, y={offset_y}. BBox center: x_in_img={center_x_in_image}, y_in_img={center_y_in_image}")
                break 

            pyautogui.click(click_x, click_y)
            await asyncio.sleep(0.75) 
        else:
            print(f"  CRITICAL ERROR: Could not find element '{target_label}' in stored CV data...")
            print(f"  Available g_icon_names were: {[el.get('g_icon_name', 'N/A') for el in all_detected_elements if el]}")
            break 
    
    print("\nTask sequence completed")

if __name__ == "__main__":
    if not gemini_client: # Check if client was initialized
        print("Exiting: Gemini client could not be initialized. Check API key and configuration.")
    else:
        asyncio.run(main_task())