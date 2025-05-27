document.addEventListener('DOMContentLoaded', function() {
    const tocContainer = document.querySelector('.post-toc-container');
    
    // Check if there's a post content and TOC container before proceeding
    if (!document.querySelector('.post-content') || !tocContainer) return;
    
    // Try to generate TOC, if no headings are found it will hide the TOC container
    generateTableOfContents();
    setupTOCScrollSpy();
});

function generateTableOfContents() {
    // Get all headings in the post content
    const postContent = document.querySelector('.post-content');
    if (!postContent) return;

    const headings = postContent.querySelectorAll('h2, h3, h4');
    const tocContainer = document.querySelector('.post-toc-container');
    
    // If no headings or no TOC container, hide TOC and exit
    if (headings.length === 0 || !tocContainer) {
        if (tocContainer) {
            tocContainer.style.display = 'none';
        }
        return;
    }
    
    // Ensure TOC is visible if we found headings
    tocContainer.style.display = 'block';

    const tocList = document.createElement('ul');
    tocList.className = 'toc-list';
    
    // Track headings for nesting
    let currentLevel = 0;
    let currentList = tocList;
    let listStack = [tocList];

    headings.forEach((heading, index) => {
        // Get heading level (h2 = 2, h3 = 3, etc.)
        const level = parseInt(heading.tagName.substring(1));
        
        // Add an ID to the heading if it doesn't have one
        if (!heading.id) {
            heading.id = `heading-${index}`;
        }
        
        // Handle nesting based on heading level
        if (currentLevel === 0) {
            // First heading - establish baseline
            currentLevel = level;
        } else if (level > currentLevel) {
            // Deeper level - create a nested list
            const nestedList = document.createElement('ul');
            listStack[listStack.length - 1].lastElementChild.appendChild(nestedList);
            listStack.push(nestedList);
            currentList = nestedList;
            currentLevel = level;
        } else if (level < currentLevel) {
            // Go back up to appropriate level
            while (listStack.length > 1 && currentLevel > level) {
                listStack.pop();
                currentLevel--;
            }
            currentList = listStack[listStack.length - 1];
            currentLevel = level;
        }
        
        // Create list item with link
        const listItem = document.createElement('li');
        const link = document.createElement('a');
        link.href = `#${heading.id}`;
        link.innerHTML = heading.innerHTML;
        link.className = 'toc-link';
        link.setAttribute('data-target', heading.id);
        listItem.appendChild(link);
        
        // Add to current list
        currentList.appendChild(listItem);
    });
    
    const tableOfContents = document.getElementById('table-of-contents');
    if (tableOfContents) {
        tableOfContents.innerHTML = ''; // Clear any existing content
        tableOfContents.appendChild(tocList);
    }
}

function setupTOCScrollSpy() {
    const headings = document.querySelectorAll('.post-content h2, .post-content h3, .post-content h4');
    if (headings.length === 0) return;
    
    const tocLinks = document.querySelectorAll('.toc-link');
    if (tocLinks.length === 0) return; // No TOC links, so no need to set up scroll spy
    
    // Offset for highlighting (slightly before the heading)
    const offset = 100;
    
    // Function to update active TOC item
    function updateTOC() {
        let activeHeading = null;
        
        // Find the heading that's currently in view
        headings.forEach(heading => {
            const rect = heading.getBoundingClientRect();
            if (rect.top <= offset) {
                activeHeading = heading;
            }
        });
        
        // Update active class
        tocLinks.forEach(link => {
            link.classList.remove('active');
            if (activeHeading && link.getAttribute('data-target') === activeHeading.id) {
                link.classList.add('active');
            }
        });
    }
    
    // Update on scroll
    window.addEventListener('scroll', updateTOC);
    
    // Initial update
    updateTOC();
    
    // Smooth scroll to heading when TOC link is clicked
    tocLinks.forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            const targetId = this.getAttribute('data-target');
            const targetElement = document.getElementById(targetId);
            if (targetElement) {
                window.scrollTo({
                    top: targetElement.offsetTop - 50,
                    behavior: 'smooth'
                });
            }
        });
    });
} 