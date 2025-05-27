/**
 * Theme Toggle and Preferences
 * 
 * Manages light/dark mode and lava lamp background preferences
 * Stores user preferences in localStorage and cookies for better compatibility
 */

document.addEventListener('DOMContentLoaded', function() {
  // Check for saved preferences or use defaults
  const darkMode = localStorage.getItem('darkMode') === 'true' || getCookie('darkMode') === 'true';
  const lavaLampEnabled = !(localStorage.getItem('lavaLampEnabled') === 'false' || getCookie('lavaLampEnabled') === 'false'); // Default to true if not set
  
  // Initialize the theme based on saved preferences
  if (darkMode) {
    document.body.classList.add('dark-mode');
    document.getElementById('dark-mode-toggle').checked = true;
  }
  
  // Initialize lava lamp based on saved preferences
  if (!lavaLampEnabled) {
    document.body.setAttribute('data-lava-lamp', 'disabled');
    document.getElementById('lava-lamp-toggle').checked = false;
  } else {
    document.body.removeAttribute('data-lava-lamp');
    document.getElementById('lava-lamp-toggle').checked = true;
  }
  
  // Dark mode toggle event listener
  document.getElementById('dark-mode-toggle').addEventListener('change', function() {
    if (this.checked) {
      document.body.classList.add('dark-mode');
      savePreference('darkMode', 'true');
    } else {
      document.body.classList.remove('dark-mode');
      savePreference('darkMode', 'false');
    }
  });
  
  // Lava lamp toggle event listener
  document.getElementById('lava-lamp-toggle').addEventListener('change', function() {
    const lavaContainer = document.getElementById('lavaLampContainer');
    
    if (this.checked) {
      document.body.removeAttribute('data-lava-lamp');
      savePreference('lavaLampEnabled', 'true');
      
      // Regenerate lava lamp if it doesn't exist
      if (!lavaContainer || lavaContainer.children.length === 0) {
        // Reload the page to reinitialize lava lamp
        // This is a simple approach; a more complex one would be to reinitialize without reload
        window.location.reload();
      }
    } else {
      document.body.setAttribute('data-lava-lamp', 'disabled');
      savePreference('lavaLampEnabled', 'false');
      
      // Remove lava lamp container if it exists
      if (lavaContainer) {
        lavaContainer.innerHTML = '';
      }
    }
  });
});

// Save preference to both localStorage and cookies
function savePreference(name, value) {
  // Save to localStorage
  localStorage.setItem(name, value);
  
  // Save to cookies as well for better compatibility
  setCookie(name, value, 365);
}

// Cookie helper functions
function setCookie(name, value, days) {
  const date = new Date();
  date.setTime(date.getTime() + (days * 24 * 60 * 60 * 1000));
  const expires = "expires=" + date.toUTCString();
  document.cookie = name + "=" + value + ";" + expires + ";path=/";
}

function getCookie(name) {
  const cookieName = name + "=";
  const cookies = document.cookie.split(';');
  for (let i = 0; i < cookies.length; i++) {
    let cookie = cookies[i].trim();
    if (cookie.indexOf(cookieName) === 0) {
      return cookie.substring(cookieName.length, cookie.length);
    }
  }
  return "";
} 