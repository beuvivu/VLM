(() => {
  let theme;
  try { theme = localStorage.getItem('vlm-theme'); } catch { /* Private browsing still works. */ }
  if (theme !== 'light' && theme !== 'dark') theme = matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
  document.documentElement.dataset.theme = theme;
  document.addEventListener('DOMContentLoaded', () => {
    document.querySelectorAll('.theme-toggle').forEach(button => {
      const label = () => button.setAttribute('aria-label', `Chuyển giao diện ${document.documentElement.dataset.theme === 'dark' ? 'sáng' : 'tối'}`);
      label();
      button.addEventListener('click', () => {
        const next = document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark';
        document.documentElement.dataset.theme = next;
        try { localStorage.setItem('vlm-theme', next); } catch { /* Theme remains usable without storage. */ }
        label();
      });
    });
    const menu = document.querySelector('.menu-toggle');
    if (menu) {
      const close = () => { menu.setAttribute('aria-expanded', 'false'); document.querySelector('#main-nav').classList.remove('open'); };
      menu.addEventListener('click', () => {
        const open = menu.getAttribute('aria-expanded') !== 'true';
        menu.setAttribute('aria-expanded', String(open));
        document.querySelector('#main-nav').classList.toggle('open', open);
      });
      document.querySelectorAll('#main-nav a').forEach(link => link.addEventListener('click', close));
      document.addEventListener('keydown', event => { if (event.key === 'Escape') close(); });
    }
  });
})();
