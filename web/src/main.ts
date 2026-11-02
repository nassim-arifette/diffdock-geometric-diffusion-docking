import './style.css';
import { loadCase } from './mol';
import { initDof, initHero, initPilot, initPool, initReverse } from './figs';
import { whenVisible } from './util';

document.getElementById('theme-toggle')?.addEventListener('click', () => {
  const root = document.documentElement;
  const dark = root.dataset.theme ? root.dataset.theme === 'dark' : matchMedia('(prefers-color-scheme: dark)').matches;
  root.dataset.theme = dark ? 'light' : 'dark';
  try { localStorage.setItem('theme', root.dataset.theme); } catch { /* storage unavailable */ }
  window.dispatchEvent(new Event('themechange'));
});

function failed(id: string, err: unknown) {
  console.error(err);
  const fig = document.getElementById(id);
  if (!fig) return;
  const p = document.createElement('p');
  p.className = 'small';
  p.textContent = 'This figure could not load. Serve the site over HTTP (for example with npm run preview) and make sure WebGL is enabled.';
  fig.prepend(p);
}
const guard = (id: string, f: () => void) => { try { f(); } catch (e) { failed(id, e); } };

guard('fig-pilot', initPilot);
loadCase().then((data) => {
  guard('top', () => initHero(data));
  const lazy: [string, () => void][] = [
    ['fig-dof', () => initDof(data)],
    ['fig-reverse', () => initReverse(data)],
    ['fig-pool', () => initPool(data)],
  ];
  for (const [id, init] of lazy) whenVisible(document.getElementById(id)!, () => guard(id, init));
}).catch((e) => ['top', 'fig-dof', 'fig-reverse', 'fig-pool'].forEach((id) => failed(id, e)));
