# HiggsArt: Voice-Controlled Image Editing
Here is the demo video:
[![HiggsArt Demo Video](https://img.youtube.com/vi/exdR0Wp0bSk/0.jpg)](https://www.youtube.com/watch?v=exdR0Wp0bSk)

HiggsArt is an innovative image editing application that replaces traditional graphical user interfaces with a powerful, intuitive voice-controlled workflow. It allows users to perform a wide range of edits, from simple transformations to complex generative AI modifications, purely through spoken commands.

This project was developed for the Boson AI Hackathon, where it was recognized as a Top 16 finalist.

## The Problem

Traditional photo editing software like Photoshop or Pixlr can have steep learning curves and complex interfaces. For users who are not graphic design experts, or for those who prefer a faster, more iterative workflow, these tools can be time-consuming and unintuitive. HiggsArt was built to bridge this gap, making image editing accessible to everyone, regardless of their technical skill, by leveraging the power of natural language.

## Key Features

HiggsArt combines standard image processing techniques with cutting-edge generative AI, all accessible through a seamless voice interface.

-   **Voice-First Interface**: Control the entire editing process without touching your mouse or keyboard. As soon as the application is running, it's ready for your voice commands.
-   **Hybrid Editing Model**: The system intelligently distinguishes between simple, native operations and complex, generative tasks based on your intent.
-   **Native Operations**: Fast, client-side execution for standard image manipulations, including:
    -   Rotating and flipping
    -   Resizing
    -   Adjusting saturation and sharpness
    -   Converting to grayscale
-   **Generative AI Operations**: Leverages powerful AI models for complex edits, such as:
    -   **In-painting**: Changing the color of objects, modifying textures, or altering specific parts of an image without regenerating the entire scene.
    -   **Object Manipulation**: Adding, removing, or replacing objects within an image.
    -   **Background Removal & Replacement**: Seamlessly segmenting and changing the background.
-   **Multi-Image Composition**: Combine elements from multiple images. For example, "take the crown from this image and put it on the man in that image."
-   **Intelligent Segmentation**: Use natural language to zoom into or crop specific objects (e.g., "zoom into the man's face").
-   **Multi-Step Command Handling**: Chain multiple commands together in a single sentence (e.g., "Flip this image horizontally, rotate it 90 degrees, and then add a hat on the man."). The system parses and executes them in sequence.
-   **Unlimited Undo/Revert**: An intelligent backup system allows you to revert one or multiple steps, giving you the freedom to experiment without fear of losing your work.
-   **Real-time Web UI**: A clean web interface displays the selected image and reflects changes in real-time as you issue commands.

## How It Works

HiggsArt is built on a distributed architecture that separates the user interface, backend logic, and AI processing into distinct components that communicate in real-time.

1.  **Frontend (index.html)**: A simple HTML/CSS/JS single-page application that serves as the visual interface. It displays the image gallery and the currently edited image. It communicates with the backend via WebSockets to receive real-time updates.

2.  **Backend Server (server.js)**: A Node.js and Express server that manages the core application state. It serves the frontend, provides an API for managing image files (list, upload, delete), and runs a WebSocket server that acts as a central hub, connecting the browser UI with the Python controller.

3.  **Terminal Controller (terminal_controller.py)**: This is the brain of the application. The Python script connects to the Node.js WebSocket server and handles all the core logic:
    -   **Audio Processing**: Captures microphone input, detects speech, and sends the audio to the Boson AI API for transcription.
    -   **Intent Parsing**: The transcribed text is sent to an Anthropic Claude model, which interprets the natural language command and converts it into a structured JSON array of editing steps.
    -   **Action Execution**: The controller iterates through the JSON instructions. Native actions are executed locally using the Pillow library. Generative actions are sent as API calls to a separate, dedicated GPU server.

4.  **AI Models**:
    -   **Audio-to-Text**: Boson AI for high-accuracy speech transcription.
    -   **Natural Language Understanding**: Anthropic Claude for parsing user intent into actionable commands.
    -   **Generative Imaging**: A separate GPU server (not included in this repository) hosts diffusion models responsible for generative tasks like in-painting, segmentation, and multi-image composition.

## Setup and Installation

To run HiggsArt, you will need Node.js and Python installed on your system.

**1. Prerequisites**
-   Node.js (v14 or later)
-   Python (v3.8 or later)
-   An account with Boson AI and Anthropic to obtain API keys.

**2. Clone the Repository**
```bash
git clone https://github.com/your-username/HiggsArt.git
cd HiggsArt
```

**3. Install Dependencies**

-   **Node.js Server**:
    ```bash
    npm install
    ```
-   **Python Controller**:
    ```bash
    pip install -r requirements_python.txt
    ```

**4. Environment Variables**
Create a `.env` file in the root directory of the project and add your API keys:
```
ANTHROPIC_API_KEY="your-anthropic-api-key"
BOSON_API_KEY= your-boson-api-key"
```
**5. Running the Application**

You need to start the three main components. It is recommended to run each command in a separate terminal window.

-   **Start the Node.js Server**:
    ```bash
    npm start
    ```
    This will start the web server on `http://localhost:3000`.

-   **Start the Python Terminal Controller**:
    ```bash
    python terminal_controller.py
    ```
    This will connect to the Node.js server and wait for commands.

-   **(Optional) Start the GPU Server**:
    For full functionality, you need to run the generative AI server. The controller expects this server to be running on `http://localhost:6000`. The code for this server is included in this repository under `qwen_inference/`.

## Usage

1.  After starting the server and controller, open your web browser and navigate to `http://localhost:3000`.
2.  The web page will display a list of images from the `/test_images` directory. You can upload your own images by dragging and dropping them onto the sidebar.
3.  In the terminal where you ran `terminal_controller.py`, type `voice` and press Enter to enable voice control.
4.  Begin speaking your commands.

**Example Commands:**
-   "Open frog."
-   "Rotate this image by 90 degrees."
-   "Increase the saturation by 30 percent."
-   "Make the frog purple."
-   "Revert this image."
-   "Zoom into the man's face."
-   "Remove the background."
