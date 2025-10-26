# Image Viewer with Voice Control

A complete image viewer system with voice-controlled image editing capabilities.

## Features

- **Web Interface**: Modern HTML/CSS/JS interface with live image reloading
- **Voice Control**: Natural language image editing commands
- **Image Editing**: Native PIL operations (rotate, flip, grayscale, resize, etc.)
- **Real-time Updates**: Live thumbnail and image reloading
- **WebSocket Communication**: Real-time sync between terminal and browser
- **File Management**: Upload, delete, and organize images

## Quick Start

### 1. Install Dependencies
```bash
# Install Node.js dependencies
npm install

# Install Python dependencies
pip install -r requirements_python.txt
```

### 2. Set Environment Variables
```bash
# Copy the example environment file
cp env.example .env

# Edit .env with your API keys
# ANTHROPIC_API_KEY=your_anthropic_key_here
# BOSON_API_KEY=your_boson_key_here

# Or set them directly
export ANTHROPIC_API_KEY="your_anthropic_key_here"
export BOSON_API_KEY="your_boson_key_here"
```

### 3. Start the System
```bash
# Terminal 1: Start the web server
npm start

# Terminal 2: Start the voice controller
python3 terminal_controller.py
```

### 4. Open the Web Interface
Open: http://localhost:3000

## Usage

### Web Interface
- **Upload Images**: Drag & drop or click to upload
- **Select Images**: Click on thumbnails to view
- **Live Updates**: Images and thumbnails reload automatically
- **Keyboard Shortcuts**: R = Manual reload, S = Toggle auto-reload

### Voice Control
1. **Enable Voice**: Type `voice` in terminal or say "voice"
2. **Basic Commands**: "list", "current", "revert", "open filename"
3. **Image Editing**: Say natural language commands like:
   - "rotate the image 90 degrees"
   - "make it grayscale"
   - "flip it vertically"
   - "increase saturation by 1.5"
   - "resize to 800 by 600"

## Voice Commands

### Basic Commands
- "list" - List available images
- "current" - Show currently selected image
- "revert" - Restore image from backup
- "open [filename]" - Select an image
- "help" - Show available commands
- "quit" - Exit the controller
- "kill" - Force kill the controller

### Image Editing Commands
- "rotate the image 90 degrees"
- "make it grayscale"
- "flip it vertically"
- "increase saturation by 1.5"
- "resize to 800 by 600"
- "make it sharper"

## File Structure

- `server.js` - Node.js server with WebSocket support
- `index.html` - Web interface with live reloading
- `terminal_controller.py` - Python voice controller
- `package.json` - Node.js dependencies
- `requirements_python.txt` - Python dependencies

## API Endpoints

- `GET /api/images` - List available images
- `GET /images/:filename` - Serve image files
- `POST /api/upload` - Upload new images
- `DELETE /api/images/:filename` - Delete images
- `GET /api/health` - Health check

## WebSocket Events

- `terminal_connect` - Terminal connection
- `browser_connect` - Browser connection
- `image_selected` - Image selection
- `get_current_image` - Get current image
- `list_images` - List images

## Image Directory

Images are stored in: `/Users/aditmeh/Desktop/test_images`

The system automatically filters out `_old` backup files from the file browser.
