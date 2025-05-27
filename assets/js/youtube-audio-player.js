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
    
    // Set thumbnail image if container exists
    const thumbnailContainer = document.getElementById('youtube-thumbnail');
    if (thumbnailContainer) {
        thumbnailContainer.style.backgroundImage = `url(https://img.youtube.com/vi/${videoId}/mqdefault.jpg)`;
    }
    
    // Set link URL for the thumbnail
    const youtubeLink = document.getElementById('youtube-link');
    if (youtubeLink) {
        youtubeLink.href = `https://www.youtube.com/watch?v=${videoId}`;
    }
    
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
    player.setVolume(defaultVolume);
    
    // Check if music was playing before page navigation
    if (sessionStorage.getItem('youtubePlayerPlaying') === 'true') {
        // Use stored time if available, otherwise use the configured start time
        const youtubeContainer = document.getElementById('youtube-container');
        const configStartTime = parseInt(youtubeContainer.dataset.startTime || 0);
        const startTime = parseFloat(sessionStorage.getItem('youtubePlayerTime')) || configStartTime;
        
        player.seekTo(startTime, true);
        player.playVideo();
        updateToggleButton('Pause Music');
        showThumbnail(true);
    }
}

function onPlayerStateChange(event) {
    // Update play/pause button based on player state
    if (event.data === YT.PlayerState.PLAYING) {
        updateToggleButton('Pause Music');
        showThumbnail(true);
    } else if (event.data === YT.PlayerState.PAUSED || event.data === YT.PlayerState.ENDED) {
        updateToggleButton('Play Music');
        showThumbnail(false);
    }
}

function showThumbnail(show) {
    const thumbnailContainer = document.getElementById('youtube-thumbnail');
    if (thumbnailContainer) {
        thumbnailContainer.style.display = show ? 'block' : 'none';
    }
}

function updateToggleButton(text) {
    const musicToggle = document.getElementById('music-toggle');
    if (musicToggle) {
        musicToggle.textContent = text === 'Pause Music' ? '⏸' : '▶';
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
        musicToggle.textContent = '▶';
    }
    
    // Save state before navigating away
    window.addEventListener('beforeunload', savePlayerState);
}); 