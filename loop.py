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

# Load environment variables (for GOOGLE_API_KEY)
load_dotenv()

# Configure Gemini API
try:
    api_key = os.getenv("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key)    
except Exception as e:
    print(f"Error configuring Gemini API: {e}")
    # gemini_model = None


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
        # Using exact match as Gemini will be instructed to use exact names
        if target_text.strip().lower() == element.get('g_icon_name', '').strip().lower():
            return element
    return None

async def launch_and_prepare_calculator():
    """Tries to launch, find, activate, and maximize the Calculator."""
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
            calculator_window.activate()
            time.sleep(0.2)
            if not calculator_window.isMaximized:
                calculator_window.maximize()
            time.sleep(0.5) 
            
            if calculator_window.isMaximized:
                print("  Calculator window maximized.")
                return calculator_window
            else:
                print("  Warning: Could not confirm Calculator window is maximized.")
                return calculator_window 
        else:
            print("Error: Calculator window not found after launch.")
            return None
    except Exception as e:
        print(f"Error during Calculator launch/preparation: {e}")
        return None

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
        response = await client.aio.models.generate_content(
            model="gemini-2.0-flash",
            contents=system_prompt
            )
        
        # if not response.parts:
        #     print("Error: Gemini returned an empty response.")
        #     return None
            
        action_sequence_str = response.text
        print(f"   Gemini Raw Response: {action_sequence_str[:200]}...") # type: ignore # Log snippet

        cleaned_response_str = clean_gemini_response(action_sequence_str) # type: ignore

        try:
            action_sequence = json.loads(cleaned_response_str).get("action_sequence", [])
        except json.JSONDecodeError as e:
            print(f"Error: Gemini response was not valid JSON. Error: {e}")
            print(f"Cleaned response attempt: {cleaned_response_str}")
            return None

        if not isinstance(action_sequence, list) or not all(isinstance(item, str) for item in action_sequence):
            print("Error: Gemini response was not a list of strings.")
            print(f"Parsed response: {action_sequence}")
            return None

        # Validate that all buttons in the sequence are from the available_buttons list
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
    calculator_window = await launch_and_prepare_calculator()
    action_sequence = None

    if calculator_window:
        print("Calculator prepared. Taking initial screenshot for CV pipeline...")
        pil_image = None
        capture_successful = False
        try:
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
                capture_successful = True 
                calculator_window = None 
        except Exception as e:
            print(f"  Error capturing specific window region: {e}. Falling back to full screen.")
            pil_image = pyautogui.screenshot() 
            capture_successful = True
            calculator_window = None 

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

    # --- Interaction Loop using the Gemini-generated action sequence ---
    print("\n▶️ Starting interaction sequence...")
    for target_label in action_sequence:
        print(f"Attempting to press: '{target_label}' using stored CV data.")

        target_element = find_element_in_cv_output(all_detected_elements, target_label)

        if target_element and 'bbox' in target_element:
            bbox = target_element['bbox'] 
            
            offset_x, offset_y = 0, 0
            if calculator_window and calculator_window.width > 0 and calculator_window.height > 0:
                offset_x = calculator_window.left
                offset_y = calculator_window.top
                # DEBUG PRINT:
                print(f"  DEBUG: Using calculator_window offsets: left={offset_x}, top={offset_y}, width={calculator_window.width}, height={calculator_window.height}")
            else:
                # DEBUG PRINT:
                print(f"  DEBUG: Not using calculator_window offsets. calculator_window is {calculator_window}")

            center_x_in_image = (bbox[0] + bbox[2]) / 2
            center_y_in_image = (bbox[1] + bbox[3]) / 2
            
            click_x = offset_x + center_x_in_image
            click_y = offset_y + center_y_in_image

            print(f"  Found '{target_label}' (exact match: {target_element.get('g_icon_name')}) at CV bbox {bbox}. Clicking at global screen coords ({click_x:.0f}, {click_y:.0f}).")
            
            # Before clicking, add a safety check for coordinates
            screen_width, screen_height = pyautogui.size()
            if not (0 <= click_x < screen_width and 0 <= click_y < screen_height):
                print(f"  CRITICAL ERROR: Calculated click coordinates ({click_x:.0f}, {click_y:.0f}) are outside screen bounds ({screen_width}x{screen_height}). Skipping click.")
                print(f"    Offsets: x={offset_x}, y={offset_y}. BBox center: x_in_img={center_x_in_image}, y_in_img={center_y_in_image}")
                # Optionally, you could try to re-acquire the calculator window here as a fallback,
                # or simply break the loop. For now, let's just report and break.
                break 

            pyautogui.click(click_x, click_y)
            await asyncio.sleep(0.75) 
        else:
            print(f"  CRITICAL ERROR: Could not find element '{target_label}' in stored CV data, even though Gemini was supposed to use this list.")
            print(f"  This indicates a mismatch or an issue with element finding. Stopping task.")
            print(f"  Available g_icon_names were: {[el.get('g_icon_name', 'N/A') for el in all_detected_elements if el]}")
            break 
    
    print("\nTask sequence completed")

if __name__ == "__main__":
    # if not gemini_model:
    #     print("Exiting: Gemini model could not be initialized. Check API key and configuration.")
    # else:
    asyncio.run(main_task())