/** Pure geometry and file helpers, no browser APIs (tested in test/geometry.test.ts). */
export type V3 = [number, number, number];
export interface Pose {
  rank: number; confidence: number; rmsd: number; valid: boolean; failed_checks: string[]; checks: Record<string, boolean>;
  molblock: string; frames: V3[][]; frame_kinds: string[]; frame_steps: (number | null)[];
  frame_rmsd: number[]; frame_min_distance: number[]; clash_atoms: number[]; min_distance: number;
}
export interface Case {
  complex: string; run_id: string; backends: Record<string, string | null>; inference: Record<string, string>;
  clash_threshold: number; pocket_center: V3; protein_pdb: string;
  reference: { molblock: string; smiles: string; min_distance: number; symbols: string[]; bonds: [number, number][]; torsions: { u: number; v: number; moving: number[] }[] };
  poses: Pose[];
}

/** Coordinates of the atom block of a V2000 molblock. */
export function molCoords(molblock: string): V3[] {
  const lines = molblock.split('\n');
  const n = parseInt(lines[3].slice(0, 3), 10);
  return lines.slice(4, 4 + n).map((l) => [parseFloat(l.slice(0, 10)), parseFloat(l.slice(10, 20)), parseFloat(l.slice(20, 30))] as V3);
}

/** Same molecule, new coordinates (the atom order of the frames is the molblock's heavy-atom order). */
export function withCoords(molblock: string, coords: V3[]): string {
  const lines = molblock.split('\n');
  const n = parseInt(lines[3].slice(0, 3), 10);
  for (let i = 0; i < n; i++) {
    const [x, y, z] = coords[i];
    lines[4 + i] = x.toFixed(4).padStart(10) + y.toFixed(4).padStart(10) + z.toFixed(4).padStart(10) + lines[4 + i].slice(30);
  }
  return lines.join('\n');
}

/** Protein heavy atoms near the pocket, for distance computations in the browser. */
export function proteinAtoms(pdb: string, center: V3, radius = 26): Float64Array {
  const out: number[] = [];
  for (const l of pdb.split('\n')) {
    if (!l.startsWith('ATOM') || l.slice(76, 78).trim() === 'H') continue;
    const x = parseFloat(l.slice(30, 38)), y = parseFloat(l.slice(38, 46)), z = parseFloat(l.slice(46, 54));
    if (Math.hypot(x - center[0], y - center[1], z - center[2]) <= radius) out.push(x, y, z);
  }
  return new Float64Array(out);
}

export function minDistance(coords: V3[], protein: Float64Array): number {
  let best = Infinity;
  for (const [x, y, z] of coords) {
    for (let j = 0; j < protein.length; j += 3) {
      const d = (x - protein[j]) ** 2 + (y - protein[j + 1]) ** 2 + (z - protein[j + 2]) ** 2;
      if (d < best) best = d;
    }
  }
  return Math.sqrt(best);
}

export function rmsd(a: V3[], b: V3[]): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += (a[i][0] - b[i][0]) ** 2 + (a[i][1] - b[i][1]) ** 2 + (a[i][2] - b[i][2]) ** 2;
  return Math.sqrt(s / a.length);
}

export function centroid(c: V3[]): V3 {
  const s: V3 = [0, 0, 0];
  for (const p of c) { s[0] += p[0]; s[1] += p[1]; s[2] += p[2]; }
  return [s[0] / c.length, s[1] / c.length, s[2] / c.length];
}

/** Rodrigues rotation of points about an axis through an origin. */
export function rotate(points: V3[], axis: V3, angle: number, origin: V3, only?: Set<number>): V3[] {
  const n = Math.hypot(...axis);
  const [kx, ky, kz] = [axis[0] / n, axis[1] / n, axis[2] / n];
  const c = Math.cos(angle), s = Math.sin(angle);
  return points.map((p, i) => {
    if (only && !only.has(i)) return p;
    const x = p[0] - origin[0], y = p[1] - origin[1], z = p[2] - origin[2];
    const dot = kx * x + ky * y + kz * z;
    const cx = ky * z - kz * y, cy = kz * x - kx * z, cz = kx * y - ky * x;
    return [
      origin[0] + x * c + cx * s + kx * dot * (1 - c),
      origin[1] + y * c + cy * s + ky * dot * (1 - c),
      origin[2] + z * c + cz * s + kz * dot * (1 - c),
    ];
  });
}


/** Residue numbers, per chain, with any heavy atom within `cutoff` of the given points. */
export function residuesNear(pdb: string, points: V3[], cutoff: number): Map<string, number[]> {
  const out = new Map<string, Set<number>>();
  const c2 = cutoff * cutoff;
  for (const l of pdb.split('\n')) {
    if (!l.startsWith('ATOM') || l.slice(76, 78).trim() === 'H') continue;
    const x = parseFloat(l.slice(30, 38)), y = parseFloat(l.slice(38, 46)), z = parseFloat(l.slice(46, 54));
    if (!points.some((p) => (p[0] - x) ** 2 + (p[1] - y) ** 2 + (p[2] - z) ** 2 <= c2)) continue;
    const chain = l.slice(21, 22).trim();
    if (!out.has(chain)) out.set(chain, new Set());
    out.get(chain)!.add(parseInt(l.slice(22, 26), 10));
  }
  return new Map([...out].map(([k, v]) => [k, [...v]]));
}
