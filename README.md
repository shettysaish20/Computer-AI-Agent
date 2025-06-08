<!-- filepath: d:\Projects\EAG_V1\Session13_Assignment\ComputerAgent1\README.md -->
# VisionCraft Agent 🤖: AI-Powered Desktop GUI Automation

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/)
[![AI Agent](https://img.shields.io/badge/AI-Agent-brightgreen)](https://github.com/your-repo/VisionCraftAgent) <!-- Replace with actual repo link if available -->
[![Computer Vision](https://img.shields.io/badge/Computer%20Vision-YOLO%2FOCR-orange)](https://github.com/your-repo/VisionCraftAgent) <!-- Replace with actual repo link if available -->

VisionCraft Agent is an innovative AI system designed to understand and interact with desktop applications using cutting-edge computer vision and large language models. It can "see" an application's interface, interpret natural language commands, and execute tasks by controlling the GUI.

## ✨ Features

*   🗣️ **Natural Language Control:** Interact with applications using plain English commands.
*   👁️ **Visual UI Understanding:**
    *   **YOLO:** Detects UI elements (buttons, input fields, etc.).
    *   **OCR:** Extracts text from identified elements.
    *   **Seraphine:** Groups related UI components.
    *   **Gemini Vision:** Provides comprehensive visual analysis and icon/element naming.
*   🧠 **LLM-Powered Task Interpretation:** Google's Gemini LLM translates user commands into a sequence of UI actions based on the visual context.
*   🖱️ **Automated GUI Interaction:** Uses PyAutoGUI for precise mouse clicks and keyboard inputs.
*   🚀 **Application Management:** Automatically launches and focuses the target application (e.g., Windows Calculator) using `pygetwindow`.
*   📸 **Dynamic Screenshot Analysis:** Captures application state for real-time decision making.

## ⚙️ How It Works

The agent follows a sophisticated pipeline to achieve GUI automation:

1.  **Initialization:**
    *   The agent launches the target application (e.g., Windows Calculator) and brings it into focus.
    *   A screenshot of the application window is captured.
2.  **Computer Vision Analysis (`main.py`):**
    *   The screenshot is processed by the CV pipeline:
        *   **YOLO** detects potential UI elements and their bounding boxes.
        *   **OCR** reads text within these elements.
        *   **Seraphine** logically groups these elements.
        *   **Gemini Vision** analyzes the visual information, providing semantic names (e.g., `g_icon_name`) for elements.
    *   The output is a structured JSON containing identified UI elements with their properties (text, bounding box, `g_icon_name`).
3.  **User Command Input:**
    *   The user provides a natural language command (e.g., "Calculate 50 times 3 and then add 10").
4.  **LLM Task Interpretation (`loop.py`):**
    *   The user's command and the list of recognized UI elements (specifically their `g_icon_name`s) are sent to a Gemini LLM.
    *   A carefully crafted system prompt guides the LLM to convert the command into a JSON array of `action_sequence` (e.g., `["5", "0", "multiply", "3", "equals", "add", "1", "0", "equals"]`).
5.  **Action Execution (`loop.py`):**
    *   The agent parses the `action_sequence` from the LLM's response.
    *   For each action (button label) in the sequence:
        *   It finds the corresponding UI element from the CV pipeline's output.
        *   It calculates the center coordinates of the element's bounding box relative to the application window.
        *   **PyAutoGUI** is used to click on these coordinates, simulating user interaction.
6.  **Loop:** The agent can then await further commands or perform subsequent actions.

## 🛠️ Technologies Used

*   **Core Language:** Python 3.7+
*   **GUI Automation:** PyAutoGUI
*   **Window Management:** `pygetwindow`
*   **Image Processing:** OpenCV, Pillow
*   **LLMs:** `google-genai` (for Gemini Pro and Gemini Vision Pro)
*   **CV Pipeline Components:**
    *   YOLO (Object Detection)
    *   OCR (e.g., Tesseract, or as part of a larger vision model)
    *   Seraphine (Custom Grouping Logic)
*   **Environment Management:** `python-dotenv`
*   **Utilities:** `subprocess`

## 🚀 Getting Started

### Prerequisites

*   Python 3.7 or higher.
*   Access to Google AI Studio and a `GEMINI_API_KEY`.
*   Windows Operating System (current implementation targets Windows Calculator, MacOS/Linux Support in development)

### Installation

1.  **Clone the repository (if applicable):**
    ```bash
    git clone <repository-url>
    cd ComputerAgent1
    ```
2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    venv\Scripts\activate  # Windows
    # source venv/bin/activate  # macOS/Linux
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Set up your environment variables:**
    *   Create a `.env` file in the `ComputerAgent1` directory.
    *   Add your Google API key to the `.env` file:
        ```
        GEMINI_API_KEY="YOUR_API_KEY_HERE"
        ```

### Running the Agent

1.  Ensure the target application (Windows Calculator) is available.
2.  Execute the main agent script:
    ```bash
    python agent.py
    ```
3.  The script will launch the Calculator, perform an initial visual analysis, and then prompt you to enter commands.

    Example command: `calculate 125 plus 375 then press equals`

## 🎯 Current Target Application

*   **Windows Calculator:** The agent is currently configured and demonstrated to work with the standard Windows Calculator application.

## 🔮 Future Scope

*   Extend support to other desktop applications and web browsers.
*   Enhance the CV pipeline for greater accuracy and robustness across diverse UIs.
*   Implement more complex multi-step reasoning and action planning.
*   Improve error handling and recovery mechanisms.
*   Explore integration with `browserMCP` for advanced web automation.

## 📄 License

This project is licensed under the terms of the [MIT LICENSE](./LICENSE) file.
