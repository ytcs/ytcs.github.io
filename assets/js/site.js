// Theme override, table of contents for long pages, and click-to-enlarge figures.
// The page is complete without this script; each piece only adds to it.

(function () {
  var root = document.documentElement;
  var darkQuery = window.matchMedia('(prefers-color-scheme: dark)');

  function currentTheme() {
    return root.dataset.theme || (darkQuery.matches ? 'dark' : 'light');
  }

  function initThemeToggle() {
    var button = document.querySelector('.theme-toggle');
    if (!button) return;
    function label() {
      button.textContent = currentTheme() === 'dark' ? 'Light' : 'Dark';
    }
    button.hidden = false;
    label();
    button.addEventListener('click', function () {
      var next = currentTheme() === 'dark' ? 'light' : 'dark';
      // Matching the OS again means no override is needed.
      if (next === (darkQuery.matches ? 'dark' : 'light')) {
        delete root.dataset.theme;
        try { localStorage.removeItem('theme'); } catch (e) {}
      } else {
        root.dataset.theme = next;
        try { localStorage.setItem('theme', next); } catch (e) {}
      }
      label();
    });
    darkQuery.addEventListener('change', label);
  }

  // Only posts with at least three sections get a contents list.
  function initToc() {
    var body = document.querySelector('.post-body');
    if (!body) return;
    var headings = body.querySelectorAll('h2[id], h3[id]');
    if (body.querySelectorAll('h2[id]').length < 3) return;

    var toc = document.createElement('details');
    toc.className = 'toc';
    var summary = document.createElement('summary');
    summary.textContent = 'Contents';
    toc.appendChild(summary);

    var list = document.createElement('ol');
    var sublist = null;
    headings.forEach(function (h) {
      var item = document.createElement('li');
      var link = document.createElement('a');
      link.href = '#' + h.id;
      link.textContent = h.textContent;
      item.appendChild(link);
      if (h.tagName === 'H3' && list.lastElementChild) {
        if (!sublist) {
          sublist = document.createElement('ol');
          list.lastElementChild.appendChild(sublist);
        }
        sublist.appendChild(item);
      } else {
        sublist = null;
        list.appendChild(item);
      }
    });
    toc.appendChild(list);
    // The rail spans the article so the list can stick inside it on wide screens.
    var rail = document.createElement('div');
    rail.className = 'toc-rail';
    rail.appendChild(toc);
    body.parentNode.insertBefore(rail, body);

    // In the wide layout the list sits in the margin and stays open.
    document.querySelector('.page').classList.add('has-toc');
    var wide = window.matchMedia('(min-width: 1200px)');
    function sync() { toc.open = wide.matches; }
    sync();
    wide.addEventListener('change', sync);
  }

  function initFigureZoom() {
    var images = document.querySelectorAll('.post-body img');
    if (!images.length || typeof HTMLDialogElement !== 'function') return;

    var dialog = document.createElement('dialog');
    dialog.className = 'zoom';
    var big = document.createElement('img');
    dialog.appendChild(big);
    document.body.appendChild(dialog);
    dialog.addEventListener('click', function () { dialog.close(); });

    images.forEach(function (img) {
      img.addEventListener('click', function () {
        big.src = img.currentSrc || img.src;
        big.alt = img.alt;
        dialog.showModal();
      });
    });
  }

  // Home page: one tab per project. The selected tab follows the URL hash, so
  // links like /#quantum-gravity or /#matter open the right project.
  function initProjectTabs() {
    var container = document.querySelector('.projects');
    if (!container) return;
    var sections = Array.prototype.slice.call(container.querySelectorAll('.project'));
    if (!sections.length) return;

    var tablist = document.createElement('div');
    tablist.className = 'tabs';
    tablist.setAttribute('role', 'tablist');
    var tabs = sections.map(function (section) {
      var heading = section.querySelector('h2');
      var tab = document.createElement('button');
      tab.type = 'button';
      tab.className = 'tab';
      tab.id = section.id + '-tab';
      tab.setAttribute('role', 'tab');
      tab.setAttribute('aria-controls', section.id);
      tab.textContent = heading.textContent;
      section.setAttribute('role', 'tabpanel');
      section.setAttribute('aria-labelledby', tab.id);
      tablist.appendChild(tab);
      return tab;
    });
    container.insertBefore(tablist, container.firstChild);
    container.classList.add('tabbed');

    function select(index, updateHash) {
      sections.forEach(function (section, i) {
        var on = i === index;
        section.hidden = !on;
        tabs[i].setAttribute('aria-selected', on ? 'true' : 'false');
        tabs[i].tabIndex = on ? 0 : -1;
      });
      if (updateHash) history.replaceState(null, '', '#' + sections[index].id);
    }

    // The hash may name a project or a section inside one.
    function fromHash() {
      var id = decodeURIComponent(location.hash.slice(1));
      var target = id && document.getElementById(id);
      var panel = target && target.closest ? target.closest('.project') : null;
      var index = sections.indexOf(panel);
      return { index: index < 0 ? 0 : index, target: target };
    }

    function show(scroll) {
      var where = fromHash();
      select(where.index, false);
      if (scroll && where.target && where.target.classList.contains('home-section')) {
        where.target.scrollIntoView();
      }
    }

    tabs.forEach(function (tab, i) {
      tab.addEventListener('click', function () { select(i, true); });
      tab.addEventListener('keydown', function (e) {
        var next = e.key === 'ArrowRight' ? i + 1 : e.key === 'ArrowLeft' ? i - 1 : null;
        if (next === null) return;
        next = (next + tabs.length) % tabs.length;
        select(next, true);
        tabs[next].focus();
      });
    });
    window.addEventListener('hashchange', function () { show(true); });
    show(true);
  }

  document.addEventListener('DOMContentLoaded', function () {
    initThemeToggle();
    initProjectTabs();
    initToc();
    initFigureZoom();
  });
})();
