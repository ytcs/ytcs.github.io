document.addEventListener('DOMContentLoaded', function() {
    initTableOfContents();
});

function initTableOfContents() {
    // Get all headings in the post content
    const postContent = document.querySelector('.post-content');
    if (!postContent) return;

    const headings = postContent.querySelectorAll('h2, h3, h4');
    if (headings.length === 0) {
        // No headings, no TOC needed
        document.querySelector('.post-toc-container')?.remove();
        return;
    }

    // Get or create TOC container
    let tocContainer = document.querySelector('.post-toc-container');
    if (!tocContainer) return;

    // Create the toggle button (initially showing '>' to expand)
    const toggleButton = document.createElement('button');
    toggleButton.className = 'toc-toggle';
    toggleButton.setAttribute('aria-label', 'Expand Table of Contents');
    toggleButton.innerHTML = '<span class="toc-toggle-icon">›</span>';
    
    // Create the TOC wrapper
    const tocWrapper = document.createElement('div');
    tocWrapper.className = 'toc-wrapper';
    
    // Create the TOC header
    const tocHeader = document.createElement('div');
    tocHeader.className = 'toc-header';
    tocHeader.textContent = 'Table of Contents';
    
    // Generate the TOC list directly in the wrapper
    const tocList = generateTOCList(headings);
    tocList.className = 'toc-content toc-list'; // Combined class
    
    // Assemble the TOC structure
    tocWrapper.appendChild(tocHeader);
    tocWrapper.appendChild(tocList);
    
    // Clear existing content and add new structure
    tocContainer.innerHTML = '';
    tocContainer.appendChild(toggleButton);
    tocContainer.appendChild(tocWrapper);
    
    // Add the collapsed class by default
    tocContainer.classList.add('toc-collapsed');
    
    // Set up toggle functionality
    toggleButton.addEventListener('click', function() {
        tocContainer.classList.toggle('toc-collapsed');
        if (tocContainer.classList.contains('toc-collapsed')) {
            toggleButton.setAttribute('aria-label', 'Expand Table of Contents');
            toggleButton.querySelector('.toc-toggle-icon').textContent = '›';
        } else {
            toggleButton.setAttribute('aria-label', 'Collapse Table of Contents');
            toggleButton.querySelector('.toc-toggle-icon').textContent = '‹';
        }
    });
    
    // Set up scrollspy
    setupScrollSpy(headings);
}

function generateTOCList(headings) {
    const tocList = document.createElement('ul');
    
    // For maintaining hierarchy
    let currentLevel = 0;
    let currentList = tocList;
    let listStack = [tocList];
    
    headings.forEach((heading, index) => {
        // Get heading level
        const level = parseInt(heading.tagName.charAt(1));
        
        // Add an ID to the heading if it doesn't have one
        if (!heading.id) {
            heading.id = `heading-${index}`;
        }
        
        // Handle hierarchy
        if (currentLevel === 0) {
            // First heading, establish baseline
            currentLevel = level;
        } else if (level > currentLevel) {
            // Going deeper
            const nestedList = document.createElement('ul');
            listStack[listStack.length - 1].lastElementChild.appendChild(nestedList);
            listStack.push(nestedList);
            currentList = nestedList;
            currentLevel = level;
        } else if (level < currentLevel) {
            // Going back up
            while (listStack.length > 1 && currentLevel > level) {
                listStack.pop();
                currentLevel--;
            }
            currentList = listStack[listStack.length - 1];
            currentLevel = level;
        }
        
        // Create list item
        const listItem = document.createElement('li');
        
        // Create link
        const link = document.createElement('a');
        link.href = `#${heading.id}`;
        link.className = 'toc-link';
        link.setAttribute('data-target', heading.id);
        
        // Use innerHTML instead of textContent to preserve math formatting
        link.innerHTML = heading.innerHTML;
        
        listItem.appendChild(link);
        currentList.appendChild(listItem);
    });
    
    return tocList;
}

function setupScrollSpy(headings) {
    const tocLinks = document.querySelectorAll('.toc-link');
    if (!tocLinks.length) return;
    
    // Offset for highlighting (slightly before the heading)
    const offset = 100;
    
    function updateActiveTOCItem() {
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
    window.addEventListener('scroll', updateActiveTOCItem);
    
    // Initial update
    updateActiveTOCItem();
    
    // Smooth scroll to heading when TOC link is clicked
    tocLinks.forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            
            // Collapse TOC on mobile after clicking a link
            const tocContainer = document.querySelector('.post-toc-container');
            const isMobile = window.innerWidth <= 768;
            if (isMobile && tocContainer) {
                tocContainer.classList.add('toc-collapsed');
                const toggleBtn = tocContainer.querySelector('.toc-toggle');
                if (toggleBtn) {
                    toggleBtn.setAttribute('aria-label', 'Expand Table of Contents');
                    toggleBtn.querySelector('.toc-toggle-icon').textContent = '›';
                }
            }
            
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