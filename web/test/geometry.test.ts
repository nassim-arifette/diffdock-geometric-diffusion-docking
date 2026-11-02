import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { molCoords, rmsd, rotate, withCoords, type V3 } from '../src/geom';

const data = JSON.parse(readFileSync(new URL('../public/data/case.json', import.meta.url), 'utf8'));
const crystal: V3[] = molCoords(data.reference.molblock);
const dist = (a: V3, b: V3) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);

describe('browser geometry on the 1A0Q ligand', () => {
  it('every torsion keeps all bond lengths', () => {
    for (const t of data.reference.torsions) {
      const c = crystal;
      const moved = rotate(c, [c[t.u][0] - c[t.v][0], c[t.u][1] - c[t.v][1], c[t.u][2] - c[t.v][2]], 1.3, c[t.v], new Set(t.moving));
      for (const [a, b] of data.reference.bonds) expect(Math.abs(dist(moved[a], moved[b]) - dist(c[a], c[b]))).toBeLessThan(1e-9);
      expect(rmsd(moved, c)).toBeGreaterThan(0.01);
    }
  });
  it('the last recorded frame of every candidate is its pose, and its RMSD is the recorded one', () => {
    for (const p of data.poses) {
      const pose = molCoords(p.molblock);
      expect(rmsd(p.frames[p.frames.length - 1], pose)).toBeLessThan(0.01);
      expect(Math.abs(p.frame_rmsd[p.frame_rmsd.length - 1] - p.rmsd)).toBeLessThan(0.01);
    }
  });
  it('rewriting coordinates in a molblock round-trips', () => {
    const shifted = crystal.map((q) => [q[0] + 1, q[1], q[2]] as V3);
    expect(rmsd(molCoords(withCoords(data.reference.molblock, shifted)), shifted)).toBeLessThan(1e-4);
  });
});
