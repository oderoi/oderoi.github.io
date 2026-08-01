document.addEventListener('DOMContentLoaded', function() {

    // ============================================
    // MOBILE MENU TOGGLE
    // ============================================
    const menuToggle = document.getElementById('menu-toggle');
    const navMenu = document.getElementById('nav-menu');

    if (menuToggle && navMenu) {
        menuToggle.addEventListener('click', function(e) {
            e.preventDefault();
            navMenu.classList.toggle('nav-open');
        });
    }

    // ============================================
    // DARK MODE TOGGLE
    // ============================================
    const themeToggle = document.getElementById('theme-toggle');
    const body = document.body;

    // Load saved preference on page load
    const savedTheme = localStorage.getItem('theme');
    if (savedTheme === 'dark') {
        body.classList.add('dark-mode');
        if (themeToggle) themeToggle.textContent = '☀️';
    }

    if (themeToggle) {
        themeToggle.addEventListener('click', function() {
            body.classList.toggle('dark-mode');
            const isDark = body.classList.contains('dark-mode');
            localStorage.setItem('theme', isDark ? 'dark' : 'light');
            themeToggle.textContent = isDark ? '☀️' : '🌙';
        });
    }

    // ============================================
    // BACK TO TOP BUTTON
    // ============================================
    const backToTop = document.createElement('button');
    backToTop.id = 'back-to-top';
    backToTop.innerHTML = '↑';
    backToTop.title = 'Back to top';
    document.body.appendChild(backToTop);

    window.addEventListener('scroll', function() {
        if (window.scrollY > 300) {
            backToTop.classList.add('visible');
        } else {
            backToTop.classList.remove('visible');
        }
    });

    backToTop.addEventListener('click', function() {
        window.scrollTo({ top: 0, behavior: 'smooth' });
    });

});