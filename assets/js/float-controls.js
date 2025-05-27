document.addEventListener('DOMContentLoaded', function() {
    initFloatingControlsVisibility();
});

function initFloatingControlsVisibility() {
    // Find all the floating controls
    const tocControl = document.querySelector('.post-toc');
    const themeControl = document.querySelector('.theme-toggle-container');
    const bgmControl = document.querySelector('.music-controls');
    
    const controls = [
        { 
            element: tocControl, 
            position: 'left',
            edgeThreshold: 100
        },
        { 
            element: themeControl, 
            position: 'bottom-left',
            edgeThreshold: 70
        },
        { 
            element: bgmControl, 
            position: 'bottom-right',
            edgeThreshold: 70
        }
    ].filter(control => control.element);
    
    if (controls.length === 0) return;
    
    // Set initial state
    controls.forEach(control => {
        const element = control.element;
        element.classList.add('control-visible');
        element.classList.remove('control-hidden');
        element.dataset.visible = 'true';
        
        // Add individual timeout for each control
        control.timeout = null;
    });
    
    const visibilityDelay = 3000; // 3 seconds
    
    // Function to show a specific control
    function showControl(control) {
        const element = control.element;
        if (element.dataset.visible !== 'true') {
            element.classList.add('control-visible');
            element.classList.remove('control-hidden');
            element.dataset.visible = 'true';
        }
        
        // Reset the timer for this specific control
        clearTimeout(control.timeout);
        control.timeout = setTimeout(() => hideControl(control), visibilityDelay);
    }
    
    // Function to hide a specific control
    function hideControl(control) {
        const element = control.element;
        if (element.dataset.visible === 'true' && !element.matches(':hover')) {
            element.classList.remove('control-visible');
            element.classList.add('control-hidden');
            element.dataset.visible = 'false';
        }
    }
    
    // Check if mouse is near the edge for a specific control
    function isMouseNearControlEdge(x, y, control) {
        const width = window.innerWidth;
        const height = window.innerHeight;
        const threshold = control.edgeThreshold;
        
        switch (control.position) {
            case 'left':
                return x <= threshold;
            case 'right':
                return x >= width - threshold;
            case 'top':
                return y <= threshold;
            case 'bottom':
                return y >= height - threshold;
            case 'top-left':
                return x <= threshold && y <= threshold;
            case 'top-right':
                return x >= width - threshold && y <= threshold;
            case 'bottom-left':
                return x <= threshold && y >= height - threshold;
            case 'bottom-right':
                return x >= width - threshold && y >= height - threshold;
            default:
                return false;
        }
    }
    
    // Handle mouse movement
    document.addEventListener('mousemove', function(e) {
        controls.forEach(control => {
            if (isMouseNearControlEdge(e.clientX, e.clientY, control)) {
                showControl(control);
            } else {
                // Check if mouse is directly over this control
                const isOverThisControl = control.element.contains(document.elementFromPoint(e.clientX, e.clientY));
                
                if (!isOverThisControl) {
                    // If not near edge and not over control, set timeout to hide
                    clearTimeout(control.timeout);
                    control.timeout = setTimeout(() => hideControl(control), visibilityDelay);
                }
            }
        });
    });
    
    // Initial hide after delay
    controls.forEach(control => {
        control.timeout = setTimeout(() => hideControl(control), visibilityDelay);
    });
    
    // Handle hovering over controls
    controls.forEach(control => {
        const element = control.element;
        
        element.addEventListener('mouseenter', function() {
            clearTimeout(control.timeout);
            showControl(control);
        });
        
        element.addEventListener('mouseleave', function() {
            // Only restart the timer if we're not near the control's edge
            const rect = element.getBoundingClientRect();
            const centerX = rect.left + rect.width / 2;
            const centerY = rect.top + rect.height / 2;
            
            if (!isMouseNearControlEdge(centerX, centerY, control)) {
                clearTimeout(control.timeout);
                control.timeout = setTimeout(() => hideControl(control), visibilityDelay);
            }
        });
    });
    
    // Make specific control appear when scrolling (mainly for TOC)
    let scrollTimeout;
    document.addEventListener('scroll', function() {
        // Only show TOC when scrolling
        const tocData = controls.find(c => c.position === 'left');
        if (tocData) {
            showControl(tocData);
            
            // Hide other controls after a brief delay if mouse isn't near them
            controls.forEach(control => {
                if (control !== tocData) {
                    clearTimeout(control.timeout);
                    control.timeout = setTimeout(() => hideControl(control), 1000);
                }
            });
        }
    }, { passive: true });
} 