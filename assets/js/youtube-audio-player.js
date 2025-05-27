// YouTube IFrame API player
let player;
let isPlayerReady = false;

// Load YouTube IFrame API
function loadYouTubeAPI() {
    const tag = document.createElement('script');
    tag.src = "https://www.youtube.com/iframe_api";
    const firstScriptTag = document.getElementsByTagName('script')[0];
    firstScriptTag.parentNode.insertBefore(tag, firstScriptTag);
}

// Called automatically when YouTube API is ready
function onYouTubeIframeAPIReady() {
    const youtubeContainer = document.getElementById('youtube-container');
    const videoId = youtubeContainer.dataset.videoId;
    const startTime = parseInt(youtubeContainer.dataset.startTime || 0);
    
    player = new YT.Player('youtube-container', {
        height: '0',
        width: '0',
        videoId: videoId,
        playerVars: {
            'autoplay': 0,
            'controls': 0,
            'showinfo': 0,
            'modestbranding': 1,
            'loop': 1,
            'playlist': videoId,
            'fs': 0,
            'cc_load_policy': 0,
            'iv_load_policy': 3,
            'autohide': 0,
            'start': startTime
        },
        events: {
            'onReady': onPlayerReady,
            'onStateChange': onPlayerStateChange
        }
    });
}

function onPlayerReady(event) {
    isPlayerReady = true;
    
    // Set default volume (30%)
    const defaultVolume = 30;
    const savedVolume = parseInt(sessionStorage.getItem('youtubeVolume') || defaultVolume);
    player.setVolume(savedVolume);
    
    // Set volume slider value
    const volumeControl = document.getElementById('volume-control');
    if (volumeControl) {
        volumeControl.value = savedVolume;
    }
    
    // Check if music was playing before page navigation
    if (sessionStorage.getItem('youtubePlayerPlaying') === 'true') {
        // Use stored time if available, otherwise use the configured start time
        const youtubeContainer = document.getElementById('youtube-container');
        const configStartTime = parseInt(youtubeContainer.dataset.startTime || 0);
        const startTime = parseFloat(sessionStorage.getItem('youtubePlayerTime')) || configStartTime;
        
        player.seekTo(startTime, true);
        player.playVideo();
        updateToggleButton('Pause Music');
    }
}

function onPlayerStateChange(event) {
    // Update play/pause button based on player state
    if (event.data === YT.PlayerState.PLAYING) {
        updateToggleButton('Pause Music');
    } else if (event.data === YT.PlayerState.PAUSED || event.data === YT.PlayerState.ENDED) {
        updateToggleButton('Play Music');
    }
}

function updateToggleButton(text) {
    const musicToggle = document.getElementById('music-toggle');
    if (musicToggle) {
        musicToggle.textContent = text;
    }
}

function togglePlayback() {
    if (!isPlayerReady) return;
    
    const playerState = player.getPlayerState();
    if (playerState === YT.PlayerState.PLAYING) {
        player.pauseVideo();
    } else {
        player.playVideo();
    }
}

function setVolume(value) {
    if (!isPlayerReady) return;
    
    player.setVolume(value);
    sessionStorage.setItem('youtubeVolume', value);
}

// Save player state before navigating away
function savePlayerState() {
    if (!isPlayerReady) return;
    
    sessionStorage.setItem('youtubePlayerPlaying', player.getPlayerState() === YT.PlayerState.PLAYING);
    sessionStorage.setItem('youtubePlayerTime', player.getCurrentTime());
}

// Initialize when document is ready
document.addEventListener('DOMContentLoaded', function() {
    loadYouTubeAPI();
    
    // Add toggle button event listener
    const musicToggle = document.getElementById('music-toggle');
    if (musicToggle) {
        musicToggle.addEventListener('click', togglePlayback);
    }
    
    // Add volume control listener
    const volumeControl = document.getElementById('volume-control');
    if (volumeControl) {
        volumeControl.addEventListener('input', function() {
            setVolume(this.value);
        });
    }
    
    // Save state before navigating away
    window.addEventListener('beforeunload', savePlayerState);
}); 