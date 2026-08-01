// ============================================
// MOBILE MENU TOGGLE
// ============================================
document.addEventListener('DOMContentLoaded', function() {
    const menuIcon = document.querySelector('.menu-icon');
    const siteNav = document.querySelector('.site-nav');
    
    if (menuIcon && siteNav) {
        menuIcon.addEventListener('click', function(e) {
            e.preventDefault();
            siteNav.classList.toggle('nav-open');
        });
    }
});

// ============================================
// DARK MODE TOGGLE
// ============================================
document.addEventListener('DOMContentLoaded', function() {
    const toggleBtn = document.getElementById('theme-toggle');
    const body = document.body;
    
    // Load saved preference
    const savedTheme = localStorage.getItem('theme');
    if (savedTheme === 'dark') {
        body.classList.add('dark-mode');
    }
    
    if (toggleBtn) {
        toggleBtn.addEventListener('click', function() {
            body.classList.toggle('dark-mode');
            const isDark = body.classList.contains('dark-mode');
            localStorage.setItem('theme', isDark ? 'dark' : 'light');
        });
    }
});

// ============================================
// BACK TO TOP BUTTON
// ============================================
document.addEventListener('DOMContentLoaded', function() {
    const backToTop = document.createElement('button');
    backToTop.id = 'back-to-top';
    backToTop.innerHTML = '↑';
    backToTop.title = 'Back to top';
    document.body.appendChild(backToTop);
    
    window.addEventListener('scroll', function() {
        if (window.scrollY > 400) {
            backToTop.classList.add('visible');
        } else {
            backToTop.classList.remove('visible');
        }
    });
    
    backToTop.addEventListener('click', function() {
        window.scrollTo({ top: 0, behavior: 'smooth' });
    });
});