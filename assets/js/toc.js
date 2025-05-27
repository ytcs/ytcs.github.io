document.addEventListener('DOMContentLoaded', function() {
    generateTableOfContents();
    setupTOCScrollSpy();
});

function generateTableOfContents() {
    // Get all headings in the post content
    const postContent = document.querySelector('.post-content');
    if (!postContent) return;

    const headings = postContent.querySelectorAll('h2, h3, h4');
    if (headings.length === 0) {
        document.querySelector('.post-toc-container')?.classList.add('hidden');
        return;
    }

    const tocContainer = document.getElementById('table-of-contents');
    if (!tocContainer) return;

    // Create the list
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
        link.textContent = heading.textContent;
        link.className = 'toc-link';
        link.setAttribute('data-target', heading.id);
        listItem.appendChild(link);
        
        // Add to current list
        currentList.appendChild(listItem);
    });
    
    tocContainer.appendChild(tocList);
}

function setupTOCScrollSpy() {
    const headings = document.querySelectorAll('.post-content h2, .post-content h3, .post-content h4');
    if (headings.length === 0) return;
    
    const tocLinks = document.querySelectorAll('.toc-link');
    
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