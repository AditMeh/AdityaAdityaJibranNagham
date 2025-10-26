#!/usr/bin/env node
/**
 * Self-contained Node.js server for image viewer
 * Directly accesses filesystem to list and serve images
 */

const express = require('express');
const fs = require('fs');
const path = require('path');
const multer = require('multer');
const WebSocket = require('ws');

const app = express();
const PORT = 3000;
const testImagesPath = './test_images';

// Common image extensions
const imageExtensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.svg'];

// Configure multer for file uploads
const storage = multer.diskStorage({
    destination: function (req, file, cb) {
        // Ensure the directory exists
        if (!fs.existsSync(testImagesPath)) {
            fs.mkdirSync(testImagesPath, { recursive: true });
        }
        cb(null, testImagesPath);
    },
    filename: function (req, file, cb) {
        // Use original filename as-is
        cb(null, file.originalname);
    }
});

const upload = multer({ 
    storage: storage,
    fileFilter: function (req, file, cb) {
        const ext = path.extname(file.originalname).toLowerCase();
        if (imageExtensions.includes(ext)) {
            cb(null, true);
        } else {
            cb(new Error('Only image files are allowed!'), false);
        }
    },
    limits: {
        fileSize: 10 * 1024 * 1024 // 10MB limit
    }
});

// Serve static files (HTML, CSS, JS)
app.use(express.static(__dirname));

// API endpoint to list images in the test_images directory
app.get('/api/images', (req, res) => {
    try {
        if (!fs.existsSync(testImagesPath)) {
            return res.json([]);
        }

        const files = fs.readdirSync(testImagesPath);
        const images = files.filter(file => {
            const ext = path.extname(file).toLowerCase();
            // Filter out _old images and only include valid image extensions
            return imageExtensions.includes(ext) && !file.includes('_old');
        });

        // Sort alphabetically
        images.sort();

        res.json(images);
    } catch (error) {
        console.error('Error reading images directory:', error);
        res.status(500).json({ error: 'Could not read images directory' });
    }
});

// Serve images directly from the test_images directory
app.get('/images/:filename', (req, res) => {
    const filename = req.params.filename;
    const imagePath = path.join(testImagesPath, filename);
    
    // Security check - ensure the file is within the test_images directory
    const resolvedPath = path.resolve(imagePath);
    const resolvedTestPath = path.resolve(testImagesPath);
    
    if (!resolvedPath.startsWith(resolvedTestPath)) {
        return res.status(403).send('Access denied');
    }
    
    if (!fs.existsSync(imagePath)) {
        return res.status(404).send('Image not found');
    }
    
    res.sendFile(resolvedPath);
});

// Upload image endpoint
app.post('/api/upload', upload.single('image'), (req, res) => {
    try {
        if (!req.file) {
            return res.status(400).json({ error: 'No file uploaded' });
        }
        
        res.json({ 
            success: true, 
            filename: req.file.filename,
            originalName: req.file.originalname,
            size: req.file.size
        });
    } catch (error) {
        console.error('Upload error:', error);
        res.status(500).json({ error: 'Upload failed' });
    }
});

// Delete image endpoint
app.delete('/api/images/:filename', (req, res) => {
    try {
        const filename = req.params.filename;
        const imagePath = path.join(testImagesPath, filename);
        
        // Security check - ensure the file is within the test_images directory
        const resolvedPath = path.resolve(imagePath);
        const resolvedTestPath = path.resolve(testImagesPath);
        
        if (!resolvedPath.startsWith(resolvedTestPath)) {
            return res.status(403).json({ error: 'Access denied' });
        }
        
        if (!fs.existsSync(imagePath)) {
            return res.status(404).json({ error: 'Image not found' });
        }
        
        fs.unlinkSync(imagePath);
        res.json({ success: true, message: 'Image deleted successfully' });
    } catch (error) {
        console.error('Delete error:', error);
        res.status(500).json({ error: 'Delete failed' });
    }
});

// Health check endpoint
app.get('/api/health', (req, res) => {
    res.json({ 
        status: 'ok', 
        imagesPath: testImagesPath,
        imagesPathExists: fs.existsSync(testImagesPath)
    });
});

// Start HTTP server
const server = app.listen(PORT, () => {
    console.log('🚀 Image Viewer Server running!');
    console.log(`📁 Images directory: ${testImagesPath}`);
    console.log(`🌐 Open: http://localhost:${PORT}`);
    console.log(`📊 Health check: http://localhost:${PORT}/api/health`);
    console.log(`🔌 WebSocket: ws://localhost:${PORT}`);
    console.log('Press Ctrl+C to stop');
});

// WebSocket server for terminal communication
const wss = new WebSocket.Server({ server });
let terminalConnection = null;
let browserConnections = new Set();
let currentImage = null;
let splitMode = false;
let splitImages = null;

wss.on('connection', (ws, req) => {
    const clientIP = req.socket.remoteAddress;
    console.log(`🔌 New WebSocket connection from ${clientIP}`);
    
    ws.on('message', (message) => {
        try {
            const data = JSON.parse(message);
            handleWebSocketMessage(ws, data);
        } catch (error) {
            console.error('Invalid WebSocket message:', error);
            ws.send(JSON.stringify({ 
                type: 'error', 
                message: 'Invalid message format' 
            }));
        }
    });
    
    ws.on('close', () => {
        if (ws === terminalConnection) {
            console.log('📱 Terminal disconnected');
            terminalConnection = null;
        } else {
            browserConnections.delete(ws);
            console.log(`🌐 Browser disconnected (${browserConnections.size} remaining)`);
        }
    });
    
    ws.on('error', (error) => {
        console.error('WebSocket error:', error);
    });
});

// Function to clean up split images
function cleanupSplitImages() {
    if (splitImages) {
        splitImages.forEach(imagePath => {
            try {
                if (fs.existsSync(imagePath)) {
                    fs.unlinkSync(imagePath);
                    console.log(`🗑️ Deleted split image: ${path.basename(imagePath)}`);
                }
            } catch (error) {
                console.error(`Error deleting split image ${imagePath}:`, error);
            }
        });
        splitImages = null;
    }
    splitMode = false;
}

async function handleWebSocketMessage(ws, data) {
    if (data.type === 'terminal_connect') {
        // Terminal is connecting
        if (terminalConnection) {
            terminalConnection.close();
        }
        terminalConnection = ws;
        console.log('📱 Terminal connected');
        ws.send(JSON.stringify({ 
            type: 'connected', 
            message: 'Terminal connected successfully' 
        }));
    } else if (data.type === 'browser_connect') {
        // Browser is connecting
        browserConnections.add(ws);
        console.log(`🌐 Browser connected (${browserConnections.size} total)`);
    } else if (data.type === 'browser_image_selected') {
        // Browser selected an image
        cleanupSplitImages(); // Clean up any existing split images
        currentImage = data.image;
        console.log(`🌐 Browser selected: ${data.image}`);
    } else if (data.type === 'select_image' && ws === terminalConnection) {
        // Terminal wants to select an image
        const imageName = data.image;
        const imagePath = path.join(testImagesPath, imageName);
        
        // Check if image exists
        if (!fs.existsSync(imagePath)) {
            ws.send(JSON.stringify({ 
                type: 'error', 
                message: `Image '${imageName}' not found` 
            }));
            return;
        }
        
        // Broadcast to all browsers
        const broadcastData = {
            type: 'image_selected',
            image: imageName,
            timestamp: new Date().toISOString()
        };
        
        browserConnections.forEach(browser => {
            if (browser.readyState === WebSocket.OPEN) {
                browser.send(JSON.stringify(broadcastData));
            }
        });
        
        // Update current image
        cleanupSplitImages(); // Clean up any existing split images
        currentImage = imageName;
        
        // Confirm to terminal
        ws.send(JSON.stringify({ 
            type: 'success', 
            message: `Selected image: ${imageName}` 
        }));
        
        console.log(`📱 Terminal selected: ${imageName}`);
    } else if (data.type === 'list_images' && ws === terminalConnection) {
        // Terminal wants to list available images
        try {
            let images = [];
            
            if (fs.existsSync(testImagesPath)) {
                const files = fs.readdirSync(testImagesPath);
                images = files.filter(file => {
                    const ext = path.extname(file).toLowerCase();
                    // Filter out _old images and only include valid image extensions
                    return imageExtensions.includes(ext) && !file.includes('_old');
                });
                images.sort();
            }
            
            ws.send(JSON.stringify({ 
                type: 'image_list', 
                images: images 
            }));
            
            console.log(`📱 Terminal requested image list (${images.length} images)`);
        } catch (error) {
            ws.send(JSON.stringify({ 
                type: 'error', 
                message: 'Failed to read images directory' 
            }));
        }
    } else if (data.type === 'get_current_image' && ws === terminalConnection) {
        // Terminal wants to know the currently selected image
        if (currentImage) {
            ws.send(JSON.stringify({ 
                type: 'current_image', 
                image: currentImage 
            }));
        } else {
            // If no current image, try to get the first available image
            try {
                if (fs.existsSync(testImagesPath)) {
                    const files = fs.readdirSync(testImagesPath);
                    const images = files.filter(file => {
                        const ext = path.extname(file).toLowerCase();
                        return imageExtensions.includes(ext) && !file.includes('_old');
                    });
                    
                    if (images.length > 0) {
                        // Auto-select the first image
                        currentImage = images[0];
                        console.log(`📱 Auto-selected first image: ${currentImage}`);
                        
                        // Notify browsers of the selection
                        browserConnections.forEach(browser => {
                            if (browser.readyState === WebSocket.OPEN) {
                                browser.send(JSON.stringify({
                                    type: 'image_selected',
                                    image: currentImage,
                                    timestamp: new Date().toISOString()
                                }));
                            }
                        });
                        
                        ws.send(JSON.stringify({ 
                            type: 'current_image', 
                            image: currentImage 
                        }));
                    } else {
                        ws.send(JSON.stringify({ 
                            type: 'current_image', 
                            image: null,
                            message: 'No images found in directory' 
                        }));
                    }
                } else {
                    ws.send(JSON.stringify({ 
                        type: 'current_image', 
                        image: null,
                        message: 'Images directory not found' 
                    }));
                }
            } catch (error) {
                ws.send(JSON.stringify({ 
                    type: 'current_image', 
                    image: null,
                    message: 'Error reading images directory' 
                }));
            }
        }
        console.log(`📱 Terminal requested current image: ${currentImage || 'none'}`);
    } else if (data.type === 'image_split_complete' && ws === terminalConnection) {
        // Python has completed the image split processing
        try {
            // Clean up any existing split images first
            cleanupSplitImages();
            
            // Store the split image paths for cleanup later
            const baseName = path.basename(data.original, path.extname(data.original));
            const ext = path.extname(data.original);
            
            splitImages = [
                path.join(testImagesPath, data.red),
                path.join(testImagesPath, data.green),
                path.join(testImagesPath, data.blue)
            ];
            splitMode = true;
            
            // Broadcast to browsers to show split view
            const broadcastData = {
                type: 'image_split',
                original: data.original,
                red: data.red,
                green: data.green,
                blue: data.blue,
                timestamp: new Date().toISOString()
            };
            
            browserConnections.forEach(browser => {
                if (browser.readyState === WebSocket.OPEN) {
                    browser.send(JSON.stringify(broadcastData));
                }
            });
            
            console.log(`📱 Python completed image split: ${data.original}`);
        } catch (error) {
            console.error('Split image broadcast error:', error);
        }
    } else if (data.type === 'channel_selected' && ws === terminalConnection) {
        // Python has selected a channel and cleaned up
        try {
            // Turn off split mode
            splitMode = false;
            splitImages = null;
            
            // Broadcast to browsers to exit split view
            const broadcastData = {
                type: 'exit_split_view',
                original: data.original,
                selected_channel: data.selected_channel,
                timestamp: new Date().toISOString()
            };
            
            browserConnections.forEach(browser => {
                if (browser.readyState === WebSocket.OPEN) {
                    browser.send(JSON.stringify(broadcastData));
                }
            });
            
            console.log(`📱 Python selected channel: ${data.selected_channel} for ${data.original}`);
        } catch (error) {
            console.error('Channel selection broadcast error:', error);
        }
    } else if (data.type === 'voice_command' && ws === terminalConnection) {
        // Terminal sent a voice command - broadcast to browsers
        const broadcastData = {
            type: 'voice_command',
            command: data.command,
            timestamp: data.timestamp || new Date().toISOString()
        };
        
        browserConnections.forEach(browser => {
            if (browser.readyState === WebSocket.OPEN) {
                browser.send(JSON.stringify(broadcastData));
            }
        });
        
        console.log(`🎤 Voice command: "${data.command}"`);
    } else if (data.type === 'voice_status' && ws === terminalConnection) {
        // Terminal sent voice status update - broadcast to browsers
        const broadcastData = {
            type: 'voice_status',
            enabled: data.enabled,
            status: data.status,
            timestamp: new Date().toISOString()
        };
        
        browserConnections.forEach(browser => {
            if (browser.readyState === WebSocket.OPEN) {
                browser.send(JSON.stringify(broadcastData));
            }
        });
        
        console.log(`🎤 Voice status: ${data.status} (enabled: ${data.enabled})`);
    }
}

// Graceful shutdown
process.on('SIGINT', () => {
    console.log('\n👋 Server shutting down...');
    process.exit(0);
});
