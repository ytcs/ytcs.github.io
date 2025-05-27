document.addEventListener('DOMContentLoaded', function() {
    initFloatingControlsVisibility();
});

function initFloatingControlsVisibility() {
    // Find all the floating controls
    const tocControl = document.querySelector('.post-toc');
    const themeControl = document.querySelector('.theme-toggle-container');
    const bgmControl = document.querySelector('.music-controls'); // Using the correct class
    
    const floatingControls = [tocControl, themeControl, bgmControl].filter(Boolean);
    if (floatingControls.length === 0) return;
    
    // Set initial state
    floatingControls.forEach(control => {
        control.classList.add('control-visible');
        control.classList.remove('control-hidden');
        control.dataset.visible = 'true';
    });
    
    let timeout;
    const visibilityDelay = 3000; // 3 seconds
    
    // Function to show controls
    function showControls() {
        floatingControls.forEach(control => {
            if (control.dataset.visible !== 'true') {
                control.classList.add('control-visible');
                control.classList.remove('control-hidden');
                control.dataset.visible = 'true';
            }
        });
        
        // Reset the timer
        clearTimeout(timeout);
        timeout = setTimeout(hideControls, visibilityDelay);
    }
    
    // Function to hide controls
    function hideControls() {
        floatingControls.forEach(control => {
            if (control.dataset.visible === 'true' && !control.matches(':hover')) {
                control.classList.remove('control-visible');
                control.classList.add('control-hidden');
                control.dataset.visible = 'false';
            }
        });
    }
    
    // Check if mouse is near the corners or edges
    function isMouseNearEdge(x, y) {
        const edgeThreshold = 70; // pixels from edge (increased for better detection)
        const width = window.innerWidth;
        const height = window.innerHeight;
        
        // Check if near any edge or corner
        return (
            x <= edgeThreshold || // Left edge
            x >= width - edgeThreshold || // Right edge
            y <= edgeThreshold || // Top edge
            y >= height - edgeThreshold // Bottom edge
        );
    }
    
    // Handle mouse movement
    document.addEventListener('mousemove', function(e) {
        if (isMouseNearEdge(e.clientX, e.clientY)) {
            showControls();
        } else {
            // If mouse is directly over a control, don't hide it
            const isOverControl = floatingControls.some(control => 
                control && control.contains && control.contains(document.elementFromPoint(e.clientX, e.clientY))
            );
            
            if (!isOverControl) {
                // Reset the timer if we're not over a control
                clearTimeout(timeout);
                timeout = setTimeout(hideControls, visibilityDelay);
            }
        }
    });
    
    // Initial hide after delay
    timeout = setTimeout(hideControls, visibilityDelay);
    
    // Handle hovering over controls
    floatingControls.forEach(control => {
        if (!control) return;
        
        control.addEventListener('mouseenter', function() {
            clearTimeout(timeout);
            showControls();
        });
        
        control.addEventListener('mouseleave', function() {
            // Only restart the timer if we're not near an edge
            const rect = control.getBoundingClientRect();
            const centerX = rect.left + rect.width / 2;
            const centerY = rect.top + rect.height / 2;
            
            if (!isMouseNearEdge(centerX, centerY)) {
                clearTimeout(timeout);
                timeout = setTimeout(hideControls, visibilityDelay);
            }
        });
    });
    
    // Make sure controls appear when scrolling
    document.addEventListener('scroll', function() {
        showControls();
    }, { passive: true });
} 