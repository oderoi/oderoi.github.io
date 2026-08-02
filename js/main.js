document.addEventListener('DOMContentLoaded', function() {

  // ============================================
  // THEME TOGGLE
  // ============================================
  const themeToggle = document.getElementById('theme-toggle');
  const body = document.body;
  const savedTheme = localStorage.getItem('theme');

  if (savedTheme === 'dark') {
    body.classList.add('dark-mode');
  }

  if (themeToggle) {
    themeToggle.addEventListener('click', function(e) {
      e.stopPropagation();
      body.classList.toggle('dark-mode');
      const isDark = body.classList.contains('dark-mode');
      localStorage.setItem('theme', isDark ? 'dark' : 'light');
    });
  }

  // ============================================
  // MOBILE MENU
  // ============================================
  const menuToggle = document.getElementById('menu-toggle');
  const mobileNav = document.getElementById('mobile-nav');

  if (menuToggle && mobileNav) {
    menuToggle.addEventListener('click', function(e) {
      e.stopPropagation();
      const isOpen = mobileNav.classList.toggle('open');
      menuToggle.setAttribute('aria-expanded', isOpen);
    });

    document.addEventListener('click', function(e) {
      if (!menuToggle.contains(e.target) && !mobileNav.contains(e.target)) {
        mobileNav.classList.remove('open');
        menuToggle.setAttribute('aria-expanded', 'false');
      }
    });
  }

  // ============================================
  // SEARCH
  // ============================================
  const searchToggle = document.getElementById('search-toggle');
  const searchOverlay = document.getElementById('search-overlay');
  const searchInput = document.getElementById('search-input');
  const searchResults = document.getElementById('search-results');
  const searchClose = document.getElementById('search-close');

  let posts = [];
  let selectedIndex = -1;

  // Fetch search index — hardcoded path since baseurl is empty
  fetch('/search.json')
    .then(r => r.json())
    .then(data => { posts = data; })
    .catch(() => { posts = []; });

  function openSearch() {
    searchOverlay.classList.add('open');
    searchInput.value = '';
    searchResults.innerHTML = '';
    selectedIndex = -1;
    setTimeout(() => searchInput.focus(), 50);
    document.body.style.overflow = 'hidden';
  }

  function closeSearch() {
    searchOverlay.classList.remove('open');
    document.body.style.overflow = '';
    selectedIndex = -1;
  }

  function escapeRegex(s) {
    return s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  }

  function highlight(text, query) {
    if (!query) return text;
    const re = new RegExp('(' + escapeRegex(query) + ')', 'gi');
    return text.replace(re, '<mark>$1</mark>');
  }

  function renderResults(matches, query) {
    if (!query.trim()) {
      searchResults.innerHTML = '';
      return;
    }
    if (matches.length === 0) {
      searchResults.innerHTML = '<div class="search-no-results">No posts found</div>';
      return;
    }

    const html = matches.map(function(p, i) {
      return '<a class="search-result' + (i === 0 ? ' selected' : '') + '" href="' + p.url + '" data-index="' + i + '">' +
        '<div class="search-result-title">' + highlight(p.title, query) + '</div>' +
        '<div class="search-result-excerpt">' + highlight(p.excerpt, query) + '</div>' +
        '<div class="search-result-date">' + p.date + '</div>' +
      '</a>';
    }).join('');

    searchResults.innerHTML = html;
    selectedIndex = 0;
  }

  function performSearch(query) {
    const q = query.toLowerCase().trim();
    if (!q) {
      searchResults.innerHTML = '';
      selectedIndex = -1;
      return;
    }
    const matches = posts.filter(function(p) {
      return p.title.toLowerCase().includes(q) ||
             p.excerpt.toLowerCase().includes(q) ||
             p.content.toLowerCase().includes(q);
    }).slice(0, 8);
    renderResults(matches, query);
  }

  function updateSelection() {
    const items = searchResults.querySelectorAll('.search-result');
    items.forEach(function(item, i) {
      item.classList.toggle('selected', i === selectedIndex);
    });
    const selected = items[selectedIndex];
    if (selected) selected.scrollIntoView({ block: 'nearest' });
  }

  if (searchToggle) {
    searchToggle.addEventListener('click', openSearch);
  }
  if (searchClose) {
    searchClose.addEventListener('click', closeSearch);
  }
  if (searchOverlay) {
    searchOverlay.addEventListener('click', function(e) {
      if (e.target === searchOverlay) closeSearch();
    });
  }
  if (searchInput) {
    searchInput.addEventListener('input', function(e) {
      performSearch(e.target.value);
    });
  }

  document.addEventListener('keydown', function(e) {
    if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
      e.preventDefault();
      openSearch();
      return;
    }
    if (e.key === 'Escape' && searchOverlay.classList.contains('open')) {
      closeSearch();
      return;
    }
    if (!searchOverlay.classList.contains('open')) return;

    const items = searchResults.querySelectorAll('.search-result');

    if (e.key === 'ArrowDown') {
      e.preventDefault();
      selectedIndex = Math.min(selectedIndex + 1, items.length - 1);
      updateSelection();
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      selectedIndex = Math.max(selectedIndex - 1, 0);
      updateSelection();
    } else if (e.key === 'Enter') {
      e.preventDefault();
      if (items[selectedIndex]) {
        window.location.href = items[selectedIndex].href;
      }
    }
  });

  // ============================================
  // BACK TO TOP
  // ============================================
  const backToTop = document.getElementById('back-to-top');

  if (backToTop) {
    let ticking = false;

    window.addEventListener('scroll', function() {
      if (!ticking) {
        window.requestAnimationFrame(function() {
          if (window.scrollY > 400) {
            backToTop.classList.add('visible');
          } else {
            backToTop.classList.remove('visible');
          }
          ticking = false;
        });
        ticking = true;
      }
    });

    backToTop.addEventListener('click', function() {
      window.scrollTo({ top: 0, behavior: 'smooth' });
    });
  }

});