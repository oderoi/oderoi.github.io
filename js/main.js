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

  fetch('/search.json')
    .then(function(r) { return r.json(); })
    .then(function(data) { posts = data; })
    .catch(function() { posts = []; });

  function openSearch() {
    searchOverlay.classList.add('open');
    searchInput.value = '';
    searchResults.innerHTML = '';
    selectedIndex = -1;
    setTimeout(function() { searchInput.focus(); }, 50);
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
    var re = new RegExp('(' + escapeRegex(query) + ')', 'gi');
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

    var html = matches.map(function(p, i) {
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
    var q = query.toLowerCase().trim();
    if (!q) {
      searchResults.innerHTML = '';
      selectedIndex = -1;
      return;
    }
    // FIXED: removed p.content which doesn't exist in search.json
    var matches = posts.filter(function(p) {
      return p.title.toLowerCase().includes(q) ||
             p.excerpt.toLowerCase().includes(q);
    }).slice(0, 8);
    renderResults(matches, query);
  }

  function updateSelection() {
    var items = searchResults.querySelectorAll('.search-result');
    items.forEach(function(item, i) {
      item.classList.toggle('selected', i === selectedIndex);
    });
    var selected = items[selectedIndex];
    if (selected) selected.scrollIntoView({ block: 'nearest' });
  }

  if (searchToggle) searchToggle.addEventListener('click', openSearch);
  if (searchClose) searchClose.addEventListener('click', closeSearch);
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

    var items = searchResults.querySelectorAll('.search-result');

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
  // GLOSSARY TERM CARDS
  // ============================================
  var glossary = {};
  var preloadedImages = {};

  fetch('/glossary.json')
    .then(function(r) { return r.json(); })
    .then(function(data) {
      Object.assign(glossary, data);
      // Preload all images
      Object.keys(data).forEach(function(key) {
        var meanings = data[key].meanings || [];
        meanings.forEach(function(m, i) {
          if (m.image) {
            var img = new Image();
            img.onload = function() { preloadedImages[key + '-' + i] = true; };
            img.onerror = function() { preloadedImages[key + '-' + i] = false; };
            img.src = m.image;
          }
        });
      });
    })
    .catch(function() {});

  var termCard = document.createElement('div');
  termCard.id = 'term-card';
  termCard.innerHTML = '<div class="term-card-arrow"></div>' +
    '<div class="term-card-inner">' +
      '<div class="term-card-header">' +
        '<strong id="term-card-title"></strong>' +
        '<button class="term-card-close" aria-label="Close">×</button>' +
      '</div>' +
      '<div id="term-card-meanings"></div>' +
    '</div>';
  document.body.appendChild(termCard);

  function renderMeanings(meanings, title) {
    var container = document.getElementById('term-card-meanings');
    if (!meanings || meanings.length === 0) {
      container.innerHTML = '<div class="term-card-body">No definition available.</div>';
      return;
    }

    var html = meanings.map(function(m, i) {
      var hasImage = m.image && m.image !== '' && m.image !== 'null';
      var imgHtml = '';
      var key = title.toLowerCase().replace(/\s+/g, '_') + '-' + i;

      if (hasImage) {
        imgHtml = '<div class="term-card-image" data-img="' + key + '">' +
          '<div class="term-card-skeleton"></div>' +
          '<img class="term-card-img" src="' + m.image + '" alt="' + m.context + '" decoding="async" data-key="' + key + '">' +
        '</div>';
      }

      return '<div class="term-card-meaning">' +
        '<div class="term-card-context">' +
          '<span class="context-badge">' + m.context + '</span>' +
        '</div>' +
        imgHtml +
        '<div class="term-card-body">' + m.definition + '</div>' +
      '</div>';
    }).join('');

    container.innerHTML = html;

    // Fade in images after render
    meanings.forEach(function(m, i) {
      var key = title.toLowerCase().replace(/\s+/g, '_') + '-' + i;
      var img = container.querySelector('img[data-key="' + key + '"]');
      if (img) {
        if (preloadedImages[key] === true) {
          img.classList.add('loaded');
          var skeleton = img.parentNode.querySelector('.term-card-skeleton');
          if (skeleton) skeleton.style.opacity = '0';
        } else {
          img.onload = function() {
            img.classList.add('loaded');
            var skeleton = img.parentNode.querySelector('.term-card-skeleton');
            if (skeleton) skeleton.style.opacity = '0';
            preloadedImages[key] = true;
          };
          img.onerror = function() {
            img.parentNode.style.display = 'none';
            preloadedImages[key] = false;
          };
        }
      }
    });
  }

  function showTermCard(key, triggerEl) {
    var data = glossary[key];
    if (!data) return;

    document.getElementById('term-card-title').textContent = data.title;
    renderMeanings(data.meanings, data.title);

    termCard.classList.add('visible');

    var rect = triggerEl.getBoundingClientRect();
    var cardWidth = 340;
    var left = rect.left + window.scrollX;
    var top = rect.bottom + window.scrollY + 10;

    if (left + cardWidth > window.innerWidth - 20) {
      left = window.innerWidth - cardWidth - 20;
    }
    if (left < 10) left = 10;

    termCard.style.left = left + 'px';
    termCard.style.top = top + 'px';

    var arrow = termCard.querySelector('.term-card-arrow');
    var termCenter = rect.left + rect.width / 2;
    var arrowLeft = termCenter - left - 6;
    arrow.style.left = Math.max(12, Math.min(arrowLeft, cardWidth - 24)) + 'px';
  }

  function hideTermCard() {
    termCard.classList.remove('visible');
  }

  document.addEventListener('click', function(e) {
    var termEl = e.target.closest('.term');
    if (termEl) {
      e.stopPropagation();
      showTermCard(termEl.dataset.term, termEl);
      return;
    }
    if (e.target.closest('.term-card-close')) {
      hideTermCard();
      return;
    }
    if (!e.target.closest('#term-card')) {
      hideTermCard();
    }
  });

  // ============================================
  // BACK TO TOP
  // ============================================
  var backToTop = document.getElementById('back-to-top');

  if (backToTop) {
    var ticking = false;
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