# Self-Contained Image Viewer with Terminal Control

A Node.js-based image viewer that directly accesses your filesystem to display images with auto-reload functionality and terminal control.

## Features

- **Direct filesystem access** - Lists actual images from `/Users/aditmeh/Desktop/test_images`
- **Image previews** - Shows thumbnails in a sidebar
- **Auto-reload** - Selected image refreshes every 1 second
- **Drag & drop upload** - Upload images directly from your browser
- **Delete functionality** - Remove images with a click
- **Terminal control** - Control image selection from command line
- **Real-time communication** - WebSocket-based terminal-to-browser control
- **Self-contained** - Single Node.js server handles everything

## Setup

1. **Install Node.js dependencies:**
   ```bash
   npm install
   ```

2. **Install Python dependencies (for terminal control):**
   ```bash
   pip install -r requirements_python.txt
   ```

3. **Create your images folder:**
   ```bash
   mkdir -p /Users/aditmeh/Desktop/test_images
   ```

4. **Add some images to the folder:**
   ```bash
   # Copy your images to the test_images folder
   cp /path/to/your/images/* /Users/aditmeh/Desktop/test_images/
   ```

## Usage

### Web Interface

1. **Start the server:**
   ```bash
   npm start
   ```

2. **Open your browser:**
   Go to `http://localhost:3000`

3. **Use the web interface:**
   - Click any image in the left sidebar to display it
   - Drag & drop images to upload them
   - Hover over images and click "Delete" to remove them
   - The selected image will auto-reload every 1 second
   - Use keyboard shortcuts: R = manual reload, S = start/stop auto-reload

### Terminal Control

1. **Start the terminal controller:**
   ```bash
   python terminal_controller.py
   ```

2. **Use terminal commands:**
   ```
   > open image1.png          # Select an image in the browser
   > open photo2.jpg          # Switch to another image
   > help                     # Show available commands
   > quit                     # Exit the terminal controller
   ```

3. **Real-time updates:**
   - Commands typed in terminal instantly update the browser
   - Error messages for invalid image names
   - Success confirmations for valid selections

## Files

- `server.js` - Node.js server with filesystem access and WebSocket support
- `index.html` - Frontend with image selection, display, and WebSocket client
- `terminal_controller.py` - Python terminal controller for remote image selection
- `package.json` - Node.js dependencies
- `requirements_python.txt` - Python dependencies for terminal control

## API Endpoints

- `GET /api/images` - Returns list of images in the folder
- `GET /images/:filename` - Serves individual image files
- `POST /api/upload` - Upload new images
- `DELETE /api/images/:filename` - Delete an image
- `GET /api/health` - Server health check

## WebSocket Communication

- **Terminal Connection** - `ws://localhost:3000` for terminal control
- **Browser Connection** - Automatic WebSocket connection for real-time updates
- **Message Types:**
  - `terminal_connect` - Terminal registers with server
  - `browser_connect` - Browser registers with server
  - `select_image` - Terminal command to select an image
  - `image_selected` - Broadcast to browsers when image is selected

## Supported Image Formats

- JPG/JPEG
- PNG
- GIF
- BMP
- WebP
- SVG

## Terminal Commands

- `open <filename>` - Select an image in the browser
- `help` - Show available commands
- `quit` / `exit` - Exit the terminal controller

The server directly reads your filesystem and serves the actual images with real-time terminal control!
