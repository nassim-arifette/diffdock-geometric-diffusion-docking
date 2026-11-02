import { line, scaleLinear, select } from 'd3';
import {
  centroid, crystalStyle, ligandStyle, minDistance, molCoords, proteinAtoms, rmsd, rotate, sceneViewer, setBackground, withCoords,
  type Case, type V3,
} from './mol';
import { cssVar, onScreen, onThemeChange, reducedMotion } from './util';

const fmtA = (v: number) => `${v.toFixed(2)} Å`;

/** Hero: the real reverse diffusion of the top-ranked candidate, the crystal pose revealed at the end. */
export function initHero(data: Case) {
  const el = document.getElementById('hero-viewer')!;
  const { viewer } = sceneViewer(el, data, { surface: true });
  const pose = data.poses[0];
  const ghost = viewer.addModel(data.reference.molblock, 'sdf');
  ghost.setStyle({}, {});
  let lig = viewer.addModel(withCoords(pose.molblock, pose.frames[0]), 'sdf');
  lig.setStyle({}, ligandStyle());
  viewer.zoomTo({ model: ghost } as any);
  viewer.zoom(0.34);
  viewer.render();
  const out = (r: string) => document.querySelector<HTMLElement>(`#top [data-role=${r}]`)!;
  let i = 0, timer = 0, visible = true;
  onScreen(el, (v) => { visible = v; });

  const show = (k: number) => {
    viewer.removeModel(lig);
    lig = viewer.addModel(withCoords(pose.molblock, pose.frames[k]), 'sdf');
    lig.setStyle({}, ligandStyle());
    const last = k === pose.frames.length - 1;
    ghost.setStyle({}, last ? crystalStyle() : {});
    viewer.render();
    out('step').textContent = pose.frame_kinds[k] === 'random_start' ? 'random start' : `${k} / ${pose.frames.length - 1}${last ? ', final' : ''}`;
    out('rmsd').textContent = fmtA(pose.frame_rmsd[k]);
    out('dmin').textContent = fmtA(pose.frame_min_distance[k]);
    out('dmin').style.color = pose.frame_min_distance[k] < data.clash_threshold ? cssVar('--clash') : '';
  };
  const tick = () => {
    if (visible && !document.hidden) {
      show(i);
      i += 1;
      if (i >= pose.frames.length) { i = 0; timer = window.setTimeout(tick, 3800); return; }
    }
    timer = window.setTimeout(tick, i === 1 ? 900 : 380);
  };
  if (reducedMotion()) show(pose.frames.length - 1); else tick();
  out('replay').addEventListener('click', () => { clearTimeout(timer); i = 0; if (reducedMotion()) show(pose.frames.length - 1); else tick(); });
  onThemeChange(() => { setBackground(viewer); show(Math.max(0, i - 1)); });
  window.addEventListener('resize', () => viewer.resize());
}

/** Figure 1: translation, rotation and one torsion applied to the crystal ligand. */
export function initDof(data: Case) {
  const fig = document.getElementById('fig-dof')!;
  const { viewer } = sceneViewer(fig.querySelector<HTMLElement>('[data-role=viewer]')!, data);
  const crystal = molCoords(data.reference.molblock);
  const protein = proteinAtoms(data.protein_pdb, data.pocket_center);
  const ghost = viewer.addModel(data.reference.molblock, 'sdf');
  ghost.setStyle({}, crystalStyle());
  let lig = viewer.addModel(data.reference.molblock, 'sdf');
  const q = <T extends HTMLElement>(s: string) => fig.querySelector<T>(s)!;
  const tr = q<HTMLInputElement>('[data-ctl=tr]'), rot = q<HTMLInputElement>('[data-ctl=rot]'), tor = q<HTMLInputElement>('[data-ctl=tor]');
  const bond = q<HTMLSelectElement>('[data-ctl=bond]');
  const sym = data.reference.symbols;
  data.reference.torsions.forEach((t, k) => bond.add(new Option(`${sym[t.u]}${t.u + 1}–${sym[t.v]}${t.v + 1}, ${t.moving.length} atoms move`, String(k))));
  const DIR: V3 = [0.62, 0.48, -0.62];
  const AXIS: V3 = [0.3, 1, 0.25];

  function update() {
    const t = data.reference.torsions[+bond.value];
    let c = crystal;
    const twist = (+tor.value * Math.PI) / 180;
    if (twist) c = rotate(c, [c[t.u][0] - c[t.v][0], c[t.u][1] - c[t.v][1], c[t.u][2] - c[t.v][2]], twist, c[t.v], new Set(t.moving));
    const angle = (+rot.value * Math.PI) / 180;
    if (angle) c = rotate(c, AXIS, angle, centroid(c));
    const d = +tr.value;
    const n = Math.hypot(...DIR);
    c = c.map((p) => [p[0] + (DIR[0] / n) * d, p[1] + (DIR[1] / n) * d, p[2] + (DIR[2] / n) * d] as V3);
    viewer.removeModel(lig);
    lig = viewer.addModel(withCoords(data.reference.molblock, c), 'sdf');
    lig.setStyle({}, ligandStyle());
    viewer.setStyle({ model: lig, index: t.moving } as any, { stick: { radius: 0.27, colorscheme: 'magentaCarbon' } });
    viewer.render();
    q('[data-out=tr]').textContent = `${d.toFixed(1)} Å`;
    q('[data-out=rot]').textContent = `${rot.value}°`;
    q('[data-out=tor]').textContent = `${tor.value}°`;
    const r = rmsd(c, crystal), m = minDistance(c, protein);
    const ro = q<HTMLElement>('[data-role=readout]');
    ro.replaceChildren();
    const row = (k: string, v: string, cls = '') => {
      const dt = document.createElement('dt'); dt.textContent = k;
      const dd = document.createElement('dd'); dd.textContent = v; if (cls) dd.className = cls;
      ro.append(dt, dd);
    };
    row('RMSD to the crystal pose', fmtA(r), r < 2 ? 'good' : 'bad');
    row('Closest protein atom', fmtA(m), m < data.clash_threshold ? 'bad' : '');
  }
  for (const c of [tr, rot, tor]) c.addEventListener('input', update);
  bond.addEventListener('change', () => { tor.value = '0'; update(); });
  q('[data-ctl=reset]').addEventListener('click', () => { tr.value = '0'; rot.value = '0'; tor.value = '0'; update(); });
  onThemeChange(() => { setBackground(viewer); update(); });
  window.addEventListener('resize', () => viewer.resize());
  update();
}

/** Figure 2: scrub the recorded frames of any candidate. */
export function initReverse(data: Case) {
  const fig = document.getElementById('fig-reverse')!;
  const { viewer } = sceneViewer(fig.querySelector<HTMLElement>('[data-role=viewer]')!, data);
  const ghost = viewer.addModel(data.reference.molblock, 'sdf');
  ghost.setStyle({}, crystalStyle());
  let lig: any = null;
  const poseSel = fig.querySelector<HTMLSelectElement>('[data-ctl=pose]')!;
  const step = fig.querySelector<HTMLInputElement>('[data-ctl=step]')!;
  const out = fig.querySelector<HTMLElement>('[data-out=step]')!;
  const spark = fig.querySelector<SVGSVGElement>('[data-role=spark]')!;
  data.poses.forEach((p, i) => poseSel.add(new Option(`Rank ${p.rank} · confidence ${p.confidence.toFixed(2)} · ${p.rmsd.toFixed(2)} Å`, String(i))));
  const W = 320, H = 190, M = { l: 32, r: 8, t: 10, b: 22 };
  spark.setAttribute('viewBox', `0 0 ${W} ${H}`);
  let timer = 0;

  function drawSpark() {
    const p = data.poses[+poseSel.value];
    const svg = select(spark);
    svg.selectAll('*').remove();
    const x = scaleLinear().domain([0, p.frames.length - 1]).range([M.l, W - M.r]);
    const top = scaleLinear().domain([0, Math.max(...p.frame_rmsd) * 1.05]).range([H * 0.55, M.t]);
    const bot = scaleLinear().domain([0, Math.max(6, ...p.frame_min_distance.map((d) => Math.min(d, 12)))]).range([H - M.b, H * 0.62]);
    const mk = (vals: number[], y: (v: number) => number, color: string) =>
      svg.append('path').attr('d', line<number>().x((_, i) => x(i)).y((v) => y(v))(vals) ?? '').attr('fill', 'none').attr('stroke', color).attr('stroke-width', 2);
    svg.append('line').attr('x1', M.l).attr('x2', W - M.r).attr('y1', top(2)).attr('y2', top(2)).attr('stroke', cssVar('--crystal')).attr('stroke-dasharray', '3 3');
    svg.append('text').attr('x', W - M.r).attr('y', top(2) + 11).attr('text-anchor', 'end').text('2 Å success');
    mk(p.frame_rmsd, top, cssVar('--ligand'));
    svg.append('text').attr('x', M.l).attr('y', M.t + 8).text('RMSD to the crystal pose');
    svg.append('line').attr('x1', M.l).attr('x2', W - M.r).attr('y1', bot(data.clash_threshold)).attr('y2', bot(data.clash_threshold)).attr('stroke', cssVar('--clash')).attr('stroke-dasharray', '3 3');
    mk(p.frame_min_distance.map((d) => Math.min(d, 12)), bot, cssVar('--ink-2'));
    svg.append('text').attr('x', M.l).attr('y', H * 0.62 + 2).text('closest protein atom');
    svg.append('text').attr('x', M.l).attr('y', H - 6).text('random start');
    svg.append('text').attr('x', W - M.r).attr('y', H - 6).attr('text-anchor', 'end').text('final');
    svg.append('line').attr('class', 'cur').attr('y1', M.t).attr('y2', H - M.b).attr('stroke', cssVar('--ink')).attr('stroke-width', 1);
    (spark as any)._x = x;
  }

  function show() {
    const p = data.poses[+poseSel.value];
    const k = Math.min(+step.value, p.frames.length - 1);
    if (lig) viewer.removeModel(lig);
    lig = viewer.addModel(withCoords(p.molblock, p.frames[k]), 'sdf');
    lig.setStyle({}, ligandStyle());
    viewer.render();
    out.textContent = `${p.frame_kinds[k] === 'random_start' ? 'random start' : k === p.frames.length - 1 ? `${k}, final` : k} · RMSD ${fmtA(p.frame_rmsd[k])} · closest protein atom ${fmtA(p.frame_min_distance[k])}`;
    const x = (spark as any)._x;
    select(spark).select('.cur').attr('x1', x(k)).attr('x2', x(k));
  }

  poseSel.addEventListener('change', () => { step.max = String(data.poses[+poseSel.value].frames.length - 1); drawSpark(); show(); });
  step.addEventListener('input', () => { clearTimeout(timer); show(); });
  fig.querySelector('[data-ctl=play]')!.addEventListener('click', () => {
    clearTimeout(timer);
    const last = data.poses[+poseSel.value].frames.length - 1;
    if (+step.value >= last) step.value = '0';
    const go = () => { show(); if (+step.value < last) { step.value = String(+step.value + 1); timer = window.setTimeout(go, reducedMotion() ? 0 : 320); } else show(); };
    go();
  });
  step.max = String(data.poses[0].frames.length - 1);
  onThemeChange(() => { setBackground(viewer); drawSpark(); show(); });
  window.addEventListener('resize', () => viewer.resize());
  drawSpark();
  show();
}

const CHECK_LABEL: Record<string, string> = {
  input_connectivity_and_charge: 'same molecule as the input', input_stereochemistry: 'stereochemistry kept', sanitization: 'RDKit sanitisation',
  all_atoms_connected: 'all atoms connected', bond_lengths: 'bond lengths', bond_angles: 'bond angles', internal_steric_clash: 'no internal clash',
  aromatic_ring_flatness: 'aromatic rings flat', 'non-aromatic_ring_non-flatness': 'other rings not flat', double_bond_flatness: 'double bonds flat',
  internal_energy: 'internal energy', 'protein-ligand_maximum_distance': 'near the protein', minimum_distance_to_protein: 'distance to protein',
  volume_overlap_with_protein: 'overlap with protein',
};

/** Figure 3: the candidate pool, its confidence ranking and its physical checks. */
export function initPool(data: Case) {
  const fig = document.getElementById('fig-pool')!;
  const { viewer } = sceneViewer(fig.querySelector<HTMLElement>('[data-role=viewer]')!, data);
  const ghost = viewer.addModel(data.reference.molblock, 'sdf');
  ghost.setStyle({}, crystalStyle());
  let lig: any = null;
  let current = 0;
  const scatter = fig.querySelector<SVGSVGElement>('[data-role=scatter]')!;
  const checks = fig.querySelector<HTMLElement>('[data-role=checks]')!;
  const W = 320, H = 200, M = { l: 36, r: 10, t: 10, b: 30 };
  scatter.setAttribute('viewBox', `0 0 ${W} ${H}`);

  function draw() {
    const svg = select(scatter);
    svg.selectAll('*').remove();
    const x = scaleLinear().domain([0, Math.max(...data.poses.map((p) => p.rmsd)) + 0.6]).range([M.l, W - M.r]);
    const ys = data.poses.map((p) => p.confidence);
    const y = scaleLinear().domain([Math.min(...ys) - 0.2, Math.max(...ys) + 0.2]).range([H - M.b, M.t]);
    svg.append('rect').attr('x', x(0)).attr('y', M.t).attr('width', x(2) - x(0)).attr('height', H - M.b - M.t).attr('fill', cssVar('--crystal')).attr('fill-opacity', 0.1);
    svg.append('line').attr('x1', x(2)).attr('x2', x(2)).attr('y1', M.t).attr('y2', H - M.b).attr('stroke', cssVar('--crystal')).attr('stroke-dasharray', '3 3');
    svg.append('text').attr('x', x(0) + 4).attr('y', M.t + 11).text('success zone');
    for (const t of [0, 2, 4, 6]) svg.append('text').attr('x', x(t)).attr('y', H - M.b + 14).attr('text-anchor', 'middle').text(`${t} Å`);
    svg.append('text').attr('x', (M.l + W) / 2).attr('y', H - 2).attr('text-anchor', 'middle').text('RMSD to the crystal pose');
    svg.append('text').attr('transform', `translate(10,${(H - M.b) / 2}) rotate(-90)`).attr('text-anchor', 'middle').text('confidence');
    data.poses.forEach((p, i) => {
      const g = svg.append('g').style('cursor', 'pointer').on('click', () => { current = i; draw(); show(); });
      g.append('circle').attr('cx', x(p.rmsd)).attr('cy', y(p.confidence)).attr('r', i === current ? 9 : 7)
        .attr('fill', p.valid ? cssVar('--crystal') : cssVar('--clash')).attr('fill-opacity', i === current ? 1 : 0.65)
        .attr('stroke', i === current ? cssVar('--ink') : cssVar('--surface')).attr('stroke-width', 1.5);
      g.append('text').attr('x', x(p.rmsd)).attr('y', y(p.confidence) + 3.5).attr('text-anchor', 'middle').attr('font-size', 9).attr('font-weight', 700).style('fill', '#fff').text(String(p.rank));
    });
    // check grid
    checks.replaceChildren();
    const head = document.createElement('div'); head.className = 'n'; head.textContent = 'rank'; checks.append(head);
    data.poses.forEach((p, i) => {
      const h = document.createElement('div'); h.className = `h${i === current ? ' on' : ''}`; h.textContent = String(p.rank);
      h.addEventListener('click', () => { current = i; draw(); show(); });
      checks.append(h);
    });
    for (const [k, label] of Object.entries(CHECK_LABEL)) {
      const n = document.createElement('div'); n.className = 'n'; n.textContent = label; n.title = k; checks.append(n);
      for (const p of data.poses) {
        const c = document.createElement('div');
        c.className = `c ${p.checks[k] === false ? 'fail' : 'pass'}`;
        c.title = `rank ${p.rank}: ${label} ${p.checks[k] === false ? 'fails' : 'passes'}`;
        checks.append(c);
      }
    }
  }

  function show() {
    const p = data.poses[current];
    if (lig) viewer.removeModel(lig);
    lig = viewer.addModel(p.molblock, 'sdf');
    lig.setStyle({}, ligandStyle());
    if (p.clash_atoms.length) viewer.addStyle({ model: lig, index: p.clash_atoms } as any, { sphere: { radius: 0.42, color: cssVar('--clash'), opacity: 0.85 } });
    viewer.render();
  }
  onThemeChange(() => { setBackground(viewer); draw(); show(); });
  window.addEventListener('resize', () => viewer.resize());
  draw();
  show();
}

/** Figure 4: the four selectors and the pilot status. */
export function initPilot() {
  const fig = document.getElementById('fig-pilot')!;
  const pipe = fig.querySelector<HTMLElement>('[data-role=pipeline]')!;
  const rows: [string, string[], string][] = [
    ['A', ['20 raw candidates', 'rank by DiffDock confidence'], 'The baseline, what DiffDock would return on its own.'],
    ['B', ['20 raw candidates', 'keep physically valid', 'rank by confidence'], 'Does it help to drop the impossible poses first?'],
    ['C', ['20 raw candidates', 'keep physically valid', 'rank by GNINA CNNscore'], 'Does an independent scorer pick better?'],
    ['D', ['20 raw candidates', 'minimise each with GNINA', 'recheck validity', 'rank by CNNscore'], 'Can local refinement fix the near misses?'],
  ];
  for (const [letter, steps, why] of rows) {
    const r = document.createElement('div'); r.className = 'pipe';
    const b = document.createElement('b'); b.textContent = letter;
    const s = document.createElement('div');
    const st = document.createElement('div'); st.className = 'steps';
    steps.forEach((t, i) => {
      if (i) { const a = document.createElement('i'); a.textContent = '→'; st.append(a); }
      const sp = document.createElement('span'); sp.textContent = t; if (/GNINA|CNNscore/.test(t)) sp.className = 'gnina'; st.append(sp);
    });
    const w = document.createElement('div'); w.className = 'small'; w.style.marginTop = '0.35rem'; w.textContent = why;
    s.append(st, w);
    r.append(b, s);
    pipe.append(r);
  }
  const status = fig.querySelector<HTMLElement>('[data-role=status]')!;
  const b = document.createElement('b'); b.textContent = 'Status: running. ';
  status.append(b, 'The development split is running on a Colab GPU. Its results will go here with their manifests, raw poses and logs. After that, the held-out split gets a single run with the protocol unchanged.');
}
