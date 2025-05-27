/**
 * Lava Lamp Background Effect
 * 
 * Creates a semi-transparent lava lamp effect on the right side of the page.
 * Blobs are randomly generated on page load but remain static (no animation).
 * 
 * To disable, add data-lava-lamp="disabled" to the <body> tag,
 * or set lava_lamp.enabled to false in _config.yml.
 */

document.addEventListener('DOMContentLoaded', function() {
    // Check if lava lamp effect is disabled
    if (document.body.getAttribute('data-lava-lamp') === 'disabled') {
        return;
    }
    
    // Create container if it doesn't exist
    let container = document.getElementById('lavaLampContainer');
    if (!container) {
        container = document.createElement('div');
        container.id = 'lavaLampContainer';
        container.className = 'lava-lamp-container';
        document.body.appendChild(container);
        
        // Get opacity from data attribute or use default
        const opacity = parseFloat(document.body.getAttribute('data-lava-opacity') || 0.4);
        
        // Add CSS
        const style = document.createElement('style');
        style.textContent = `
            .lava-lamp-container {
                position: fixed;
                top: 0;
                right: 0;
                width: 30%;
                height: 100vh;
                overflow: hidden;
                pointer-events: none;
                z-index: 1;
                opacity: ${opacity};
            }
            
            .lava-blob {
                position: absolute;
                border-radius: 50%;
                filter: blur(30px);
                opacity: 0.7;
            }
        `;
        document.head.appendChild(style);
    }

    // Try to get colors from the data attribute
    let colors = [];
    try {
        const colorsAttr = document.body.getAttribute('data-lava-colors');
        if (colorsAttr) {
            colors = JSON.parse(colorsAttr);
        }
    } catch (e) {
        console.warn('Invalid lava lamp colors format:', e);
    }
    
    // Configuration
    const config = {
        numBlobs: parseInt(document.body.getAttribute('data-lava-num-blobs') || 6, 10),
        colors: colors.length > 0 ? colors : [
            '#C5B4E3', // Soft lavender
            '#B4E3C5', // Pastel mint
            '#E3C5B4', // Soft peach
            '#B4C5E3', // Pastel sky blue
            '#E3B4C5', // Soft rose
            '#D0E3B4'  // Pastel lime
        ],
        minSize: 150,
        maxSize: 300
    };

    // Initialize
    generateBlobs();

    /**
     * Generate random lava blobs
     */
    function generateBlobs() {
        // Clear any existing blobs
        container.innerHTML = '';
        
        // Create blobs
        for (let i = 0; i < config.numBlobs; i++) {
            createBlob();
        }
    }

    /**
     * Create a single blob
     */
    function createBlob() {
        const blob = document.createElement('div');
        blob.className = 'lava-blob';
        
        // Random properties
        const size = Math.random() * (config.maxSize - config.minSize) + config.minSize;
        const color = config.colors[Math.floor(Math.random() * config.colors.length)];
        
        // Random position within container
        const startLeft = Math.random() * 70 + 15; // percentage
        const startTop = Math.random() * 70 + 15;  // percentage
        
        // Apply styles
        blob.style.width = `${size}px`;
        blob.style.height = `${size}px`;
        blob.style.backgroundColor = color;
        blob.style.left = `${startLeft}%`;
        blob.style.top = `${startTop}%`;
        
        // Add to container
        container.appendChild(blob);
    }
}); 