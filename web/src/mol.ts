import * as $3Dmol from '3dmol';
import { cssVar } from './util';

import { molCoords, residuesNear, type Case } from './geom';
export * from './geom';

let promise: Promise<Case> | null = null;
export function loadCase(): Promise<Case> {
  promise ??= fetch(`${import.meta.env.BASE_URL}data/case.json`).then((r) => {
    if (!r.ok) throw new Error(`case.json: HTTP ${r.status}`);
    return r.json();
  });
  return promise;
}

/** A viewer with the receptor drawn as a light cartoon, its pocket as thin sticks and a faint surface. */
export function sceneViewer(el: HTMLElement, data: Case, opts: { surface?: boolean } = {}) {
  const viewer = $3Dmol.createViewer(el, { backgroundColor: cssVar('--surface'), antialias: true });
  const protein = viewer.addModel(data.protein_pdb, 'pdb');
  const ref = viewer.addModel(data.reference.molblock, 'sdf');
  viewer.setStyle({ model: protein }, { cartoon: { color: cssVar('--protein'), opacity: 1 } });
  const lig = molCoords(data.reference.molblock);
  for (const [chain, resi] of residuesNear(data.protein_pdb, lig, 5)) {
    viewer.addStyle({ model: protein, chain, resi } as any, { stick: { radius: 0.11, colorscheme: 'grayCarbon', opacity: 0.85 } });
  }
  if (opts.surface) {
    for (const [chain, resi] of residuesNear(data.protein_pdb, lig, 9)) {
      viewer.addSurface($3Dmol.SurfaceType.VDW, { opacity: 0.16, color: cssVar('--protein') }, { model: protein, chain, resi } as any);
    }
  }
  viewer.setStyle({ model: ref }, {});
  viewer.zoomTo({ model: ref } as any);
  viewer.zoom(0.62);
  return { viewer, protein, ref };
}

export const crystalStyle = () => ({ stick: { radius: 0.16, color: cssVar('--crystal'), opacity: 0.45 } });
export const ligandStyle = () => ({ stick: { radius: 0.2, colorscheme: 'magentaCarbon' } });

export function setBackground(viewer: any) {
  viewer.setBackgroundColor(cssVar('--surface'), 1);
  viewer.render();
}

export { $3Dmol };
