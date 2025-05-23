// Image Modal for Click-to-Enlarge Functionality
document.addEventListener('DOMContentLoaded', function() {
  // Create modal HTML structure
  const modal = document.createElement('div');
  modal.className = 'image-modal';
  modal.innerHTML = `
    <span class="image-modal-close">&times;</span>
    <img class="image-modal-content" alt="Enlarged image">
  `;
  document.body.appendChild(modal);

  const modalImg = modal.querySelector('.image-modal-content');
  const closeBtn = modal.querySelector('.image-modal-close');

  // Function to open modal
  function openModal(src, alt) {
    modal.style.display = 'block';
    modalImg.src = src;
    modalImg.alt = alt || 'Enlarged image';
    document.body.style.overflow = 'hidden'; // Prevent scrolling
  }

  // Function to close modal
  function closeModal() {
    modal.style.display = 'none';
    document.body.style.overflow = 'auto'; // Restore scrolling
  }

  // Add click event to all images in post content
  function addImageClickHandlers() {
    const images = document.querySelectorAll('.post-content img, .post img, article img');
    images.forEach(function(img) {
      // Skip if image is already clickable or very small
      if (img.closest('a') || img.width < 100 || img.height < 100) {
        return;
      }
      
      img.style.cursor = 'pointer';
      img.addEventListener('click', function() {
        openModal(this.src, this.alt);
      });
    });
  }

  // Close modal when clicking close button
  closeBtn.addEventListener('click', closeModal);

  // Close modal when clicking outside the image
  modal.addEventListener('click', function(e) {
    if (e.target === modal) {
      closeModal();
    }
  });

  // Close modal with Escape key
  document.addEventListener('keydown', function(e) {
    if (e.key === 'Escape' && modal.style.display === 'block') {
      closeModal();
    }
  });

  // Initialize image handlers
  addImageClickHandlers();

  // Re-run when new content is loaded (for dynamic content)
  const observer = new MutationObserver(function(mutations) {
    mutations.forEach(function(mutation) {
      if (mutation.type === 'childList' && mutation.addedNodes.length > 0) {
        addImageClickHandlers();
      }
    });
  });

  observer.observe(document.body, {
    childList: true,
    subtree: true
  });
}); 