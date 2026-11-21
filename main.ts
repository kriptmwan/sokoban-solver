// Sokoban solver for make.fun sokoban-solver-01-run-20261121.
//
// Strategy (written from scratch for this competition):
//  * A level decomposes into independent rooms off a 1-wide hallway. Boxes
//    never leave their room (a box pushed into the hallway can never return),
//    so each room is solved on its own, push-optimally, and the room
//    solutions are stitched together with shortest walks through the hallway.
//  * Each room is solved by push-based A*: a state is the sorted box cells
//    plus the player's reachable region (canonicalised to its lowest cell),
//    the heuristic is the min-cost assignment of boxes to goals over
//    floor BFS distances, and dead states are pruned with static dead
//    squares (single-box push reachability), frozen-box iteration and 2x2
//    block detection. A greedy best-first fallback runs if A* exhausts its
//    expansion budget. Budgets are counted in expansions, never wall time,
//    so two runs produce identical bytes.
//  * Fallbacks, in order: per-room A*, per-room greedy, whole-level push
//    A*, whole-level move A*. A level that still has no solution is answered
//    with an empty (valid, zero-point) solution rather than an illegal one;
//    every solution is replay-verified before it is written out.
import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const I = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const O = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

// Fixed direction order everywhere: up, down, left, right. Deterministic.
const DIRS = [
  { dx: 0, dy: -1, ch: 'u' },
  { dx: 0, dy: 1, ch: 'd' },
  { dx: -1, dy: 0, ch: 'l' },
  { dx: 1, dy: 0, ch: 'r' },
] as const;

// Expansion budgets (counts, not time: deterministic across runs).
const ROOM_ASTAR_CAP = 300_000;
const ROOM_GREEDY_CAP = 300_000;
const LEVEL_PUSH_CAP = 800_000;
const MOVE_ASTAR_CAP = 400_000;

type Level = {
  w: number;
  h: number;
  wall: Uint8Array; // 1 = wall
  goals: number[];
  goalSet: Uint8Array;
  boxes: number[];
  player: number;
};

function parseLevel(text: string): Level {
  const rows = text.split('\n').filter((r) => r.length > 0);
  const w = Math.max(...rows.map((r) => r.length));
  const h = rows.length;
  const wall = new Uint8Array(w * h).fill(1);
  const goalSet = new Uint8Array(w * h);
  const goals: number[] = [];
  const boxes: number[] = [];
  let player = 0;
  rows.forEach((row, y) => {
    for (let x = 0; x < row.length; x++) {
      const ch = row[x] ?? '#';
      const i = y * w + x;
      if (ch === '#') continue;
      wall[i] = 0;
      if (ch === '.' || ch === '*' || ch === '+') {
        goalSet[i] = 1;
        goals.push(i);
      }
      if (ch === '$' || ch === '*') boxes.push(i);
      if (ch === '@' || ch === '+') player = i;
    }
  });
  boxes.sort((a, b) => a - b);
  goals.sort((a, b) => a - b);
  return { w, h, wall, goals, goalSet, boxes, player };
}

// BFS distances from src over cells where allowed[i] is 1.
function bfsDist(src: number, allowed: Uint8Array, w: number, h: number): Int32Array {
  const dist = new Int32Array(w * h).fill(-1);
  if (!allowed[src]) return dist;
  const q: number[] = [src];
  dist[src] = 0;
  for (let qi = 0; qi < q.length; qi++) {
    const c = q[qi]!;
    const cx = c % w;
    const cy = (c / w) | 0;
    for (const d of DIRS) {
      const nx = cx + d.dx;
      const ny = cy + d.dy;
      if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue;
      const n = ny * w + nx;
      if (!allowed[n] || dist[n] >= 0) continue;
      dist[n] = dist[c]! + 1;
      q.push(n);
    }
  }
  return dist;
}

// Shortest walk (u/d/l/r) from `from` to `to` avoiding walls and boxes.
// Returns the move string, or null when unreachable. Deterministic.
function walkPath(
  from: number,
  to: number,
  wall: Uint8Array,
  boxSet: Uint8Array,
  w: number,
  h: number,
): string | null {
  if (from === to) return '';
  const prev = new Int32Array(w * h).fill(-2);
  const prevCh = new Array<string>(w * h).fill('');
  const q: number[] = [from];
  prev[from] = -1;
  for (let qi = 0; qi < q.length; qi++) {
    const c = q[qi]!;
    if (c === to) break;
    const cx = c % w;
    const cy = (c / w) | 0;
    for (const d of DIRS) {
      const nx = cx + d.dx;
      const ny = cy + d.dy;
      if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue;
      const n = ny * w + nx;
      if (prev[n] !== -2 || wall[n] || boxSet[n]) continue;
      prev[n] = c;
      prevCh[n] = d.ch;
      q.push(n);
    }
  }
  if (prev[to] === -2) return null;
  let s = '';
  for (let c = to; c !== from; c = prev[c]!) s = prevCh[c]! + s;
  return s;
}

// Player-reachable region from `start` over `allowed` cells, avoiding boxes.
// Returns a Uint8Array mask and the lowest cell of the region.
function regionOf(
  start: number,
  allowed: Uint8Array,
  boxSet: Uint8Array,
  w: number,
  h: number,
): { mask: Uint8Array; canon: number } {
  const mask = new Uint8Array(w * h);
  const q: number[] = [start];
  mask[start] = 1;
  let canon = start;
  for (let qi = 0; qi < q.length; qi++) {
    const c = q[qi]!;
    if (c < canon) canon = c;
    const cx = c % w;
    const cy = (c / w) | 0;
    for (const d of DIRS) {
      const nx = cx + d.dx;
      const ny = cy + d.dy;
      if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue;
      const n = ny * w + nx;
      if (mask[n] || !allowed[n] || boxSet[n]) continue;
      mask[n] = 1;
      q.push(n);
    }
  }
  return { mask, canon };
}

// Min-cost assignment of boxes to goals (exact by permutation for n <= 6,
// otherwise the sum of per-box minima, still a lower bound on pushes).
// dist[g][cell] is the BFS distance from goal g. Returns Infinity when some
// box cannot reach any goal.
function matchingHeuristic(
  boxes: number[],
  goals: number[],
  goalDist: Int32Array[],
): number {
  const n = boxes.length;
  if (n === 0) return 0;
  if (n <= 6) {
    let best = Infinity;
    const perm = goals.map((_, i) => i);
    const rec = (k: number, acc: number): void => {
      if (acc >= best) return;
      if (k === n) {
        if (acc < best) best = acc;
        return;
      }
      for (let j = k; j < n; j++) {
        [perm[k], perm[j]] = [perm[j]!, perm[k]!];
        const d = goalDist[perm[k]!]![boxes[k]!];
        if (d >= 0) rec(k + 1, acc + d);
        [perm[k], perm[j]] = [perm[j]!, perm[k]!];
      }
    };
    rec(0, 0);
    return best;
  }
  let sum = 0;
  for (const b of boxes) {
    let m = Infinity;
    for (const gd of goalDist) {
      const d = gd[b];
      if (d >= 0 && d < m) m = d;
    }
    if (m === Infinity) return Infinity;
    sum += m;
  }
  return sum;
}

type Push = { box: number; dir: number }; // dir indexes DIRS

type RoomSpec = {
  w: number;
  h: number;
  wall: Uint8Array;
  goalSet: Uint8Array;
  cells: Uint8Array; // cells the player may stand on (room + entry cell)
  boxCells: Uint8Array; // cells a box may occupy (room only)
  boxes: number[];
  goals: number[];
  entry: number; // hallway cell the player starts on
  exitRequired: boolean; // final player region must reach the entry cell
};

// Push-based search over one room (or the whole level). Returns the push
// list, or null. `greedy` switches the priority from A* (g+h) to best-first.
function pushSearch(spec: RoomSpec, cap: number, greedy: boolean): Push[] | null {
  const { w, h, wall, goalSet, cells, boxCells, boxes, goals, entry, exitRequired } = spec;
  const n = boxes.length;
  const sortedGoals = [...goals].sort((a, b) => a - b);

  // goalDist[g][cell]: BFS distance from goal g over player cells.
  const goalDist = sortedGoals.map((g) => bfsDist(g, cells, w, h));

  // Static dead cells: a box on cell c can reach a goal by single-box pushes
  // (player stance assumed reachable, other boxes ignored) iff live[c].
  // Computed by forward BFS over box positions from c.
  const live = new Uint8Array(w * h);
  for (let c = 0; c < w * h; c++) {
    if (!boxCells[c]) continue;
    if (goalSet[c]) {
      live[c] = 1;
      continue;
    }
    const seen = new Uint8Array(w * h);
    const q: number[] = [c];
    seen[c] = 1;
    let ok = false;
    for (let qi = 0; qi < q.length && !ok; qi++) {
      const b = q[qi]!;
      const bx = b % w;
      const by = (b / w) | 0;
      for (const d of DIRS) {
        const sx = bx - d.dx;
        const sy = by - d.dy;
        const dx = bx + d.dx;
        const dy = by + d.dy;
        if (sx < 0 || sy < 0 || sx >= w || sy >= h) continue;
        if (dx < 0 || dy < 0 || dx >= w || dy >= h) continue;
        const stance = sy * w + sx;
        const dest = dy * w + dx;
        if (!cells[stance] || !boxCells[dest] || seen[dest]) continue;
        if (goalSet[dest]) {
          ok = true;
          break;
        }
        seen[dest] = 1;
        q.push(dest);
      }
    }
    if (ok) live[c] = 1;
  }

  const boxSetOf = (bs: number[]): Uint8Array => {
    const s = new Uint8Array(w * h);
    for (const b of bs) s[b] = 1;
    return s;
  };

  // Dead-state test: box on a dead cell, a frozen box off its goal, or an
  // all-box 2x2 with a box off its goal.
  const isDead = (bs: number[]): boolean => {
    for (const b of bs) if (!live[b]) return true;
    const set = boxSetOf(bs);
    // Frozen boxes, iterated: a box is frozen when every push is blocked by
    // a wall or a frozen box (in either the destination or the stance).
    const frozen = new Uint8Array(w * h);
    let changed = true;
    while (changed) {
      changed = false;
      for (const b of bs) {
        if (frozen[b]) continue;
        const bx = b % w;
        const by = (b / w) | 0;
        let movable = false;
        for (const d of DIRS) {
          const sx = bx - d.dx;
          const sy = by - d.dy;
          const dx = bx + d.dx;
          const dy = by + d.dy;
          if (sx < 0 || sy < 0 || sx >= w || sy >= h) continue;
          if (dx < 0 || dy < 0 || dx >= w || dy >= h) continue;
          const stance = sy * w + sx;
          const dest = dy * w + dx;
          const destOk = !wall[dest] && !frozen[dest];
          const stanceOk = !wall[stance] && !frozen[stance];
          if (destOk && stanceOk) {
            movable = true;
            break;
          }
        }
        if (!movable) {
          frozen[b] = 1;
          changed = true;
        }
      }
    }
    for (const b of bs) if (frozen[b] && !goalSet[b]) return true;
    // 2x2 blocks of boxes.
    for (let y = 0; y + 1 < h; y++) {
      for (let x = 0; x + 1 < w; x++) {
        const a = y * w + x;
        if (set[a] && set[a + 1] && set[a + w] && set[a + w + 1]) {
          if (!goalSet[a] || !goalSet[a + 1] || !goalSet[a + w] || !goalSet[a + w + 1]) return true;
        }
      }
    }
    return false;
  };

  const solved = (bs: number[]): boolean => {
    if (bs.length !== sortedGoals.length) return false;
    for (let i = 0; i < bs.length; i++) if (bs[i] !== sortedGoals[i]) return false;
    return true;
  };

  // --- A* / greedy state store ---
  const sBoxes: number[][] = [];
  const sPlayer: number[] = []; // canonical player cell
  const sParent: number[] = [];
  const sPush: (Push | null)[] = [];
  const sG: number[] = [];
  const best = new Map<string, number>(); // key -> best g
  let seq = 0;
  // Heap entries: [priority, -goalsOn, h, seq, stateIdx]; priority = g+h or h.
  const heap: [number, number, number, number, number][] = [];
  const less = (a: [number, number, number, number, number], b: [number, number, number, number, number]): boolean => {
    if (a[0] !== b[0]) return a[0] < b[0];
    if (a[1] !== b[1]) return a[1] < b[1];
    if (a[2] !== b[2]) return a[2] < b[2];
    return a[3] < b[3];
  };
  const heapPush = (e: [number, number, number, number, number]): void => {
    let i = heap.length;
    heap.push(e);
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (!less(heap[i]!, heap[p]!)) break;
      [heap[i], heap[p]] = [heap[p]!, heap[i]!];
      i = p;
    }
  };
  const heapPop = (): [number, number, number, number, number] => {
    const top = heap[0]!;
    const last = heap.pop()!;
    if (heap.length > 0) {
      heap[0] = last;
      for (let i = 0; ; ) {
        let c = 2 * i + 1;
        if (c >= heap.length) break;
        if (c + 1 < heap.length && less(heap[c + 1]!, heap[c]!)) c++;
        if (!less(heap[c]!, heap[i]!)) break;
        [heap[i], heap[c]] = [heap[c]!, heap[i]!];
        i = c;
      }
    }
    return top;
  };

  const startBoxes = [...boxes].sort((a, b) => a - b);
  if (isDead(startBoxes)) return null;
  const h0 = matchingHeuristic(startBoxes, sortedGoals, goalDist);
  if (h0 === Infinity) return null;
  const startSet = boxSetOf(startBoxes);
  const startRegion = regionOf(entry, cells, startSet, w, h);
  const keyOf = (bs: number[], canon: number): string => bs.join(',') + '|' + canon;
  const addState = (bs: number[], canon: number, parent: number, push: Push | null, g: number): number => {
    const idx = sBoxes.length;
    sBoxes.push(bs);
    sPlayer.push(canon);
    sParent.push(parent);
    sPush.push(push);
    sG.push(g);
    best.set(keyOf(bs, canon), g);
    const hh = matchingHeuristic(bs, sortedGoals, goalDist);
    const onGoal = bs.reduce((a, b) => a + (goalSet[b] ? 1 : 0), 0);
    heapPush([greedy ? hh : g + hh, -onGoal, hh, seq++, idx]);
    return idx;
  };
  addState(startBoxes, startRegion.canon, -1, null, 0);

  let expanded = 0;
  while (heap.length > 0 && expanded < cap) {
    const [, , , , idx] = heapPop();
    const bs = sBoxes[idx]!;
    const g = sG[idx]!;
    if (best.get(keyOf(bs, sPlayer[idx]!)) !== g) continue; // stale entry
    if (solved(bs)) {
      // Reconstruct pushes.
      const pushes: Push[] = [];
      for (let t = idx; t > 0; t = sParent[t]!) pushes.push(sPush[t]!);
      pushes.reverse();
      return pushes;
    }
    expanded++;
    const set = boxSetOf(bs);
    // Player region for this state: recompute from a representative cell.
    // The canonical cell is inside the region, so flood from it.
    const region = regionOf(sPlayer[idx]!, cells, set, w, h);
    if (solved(bs) && exitRequired && !region.mask[entry]) continue;
    for (let bi = 0; bi < bs.length; bi++) {
      const b = bs[bi]!;
      const bx = b % w;
      const by = (b / w) | 0;
      for (let di = 0; di < DIRS.length; di++) {
        const d = DIRS[di]!;
        const sx = bx - d.dx;
        const sy = by - d.dy;
        const dx = bx + d.dx;
        const dy = by + d.dy;
        if (sx < 0 || sy < 0 || sx >= w || sy >= h) continue;
        if (dx < 0 || dy < 0 || dx >= w || dy >= h) continue;
        const stance = sy * w + sx;
        const dest = dy * w + dx;
        if (!region.mask[stance]) continue;
        if (!boxCells[dest] || set[dest]) continue;
        const nb = bs.slice();
        nb[bi] = dest;
        nb.sort((a, b) => a - b);
        if (isDead(nb)) continue;
        const hh = matchingHeuristic(nb, sortedGoals, goalDist);
        if (hh === Infinity) continue;
        const nSet = boxSetOf(nb);
        const nRegion = regionOf(b, cells, nSet, w, h);
        const k = keyOf(nb, nRegion.canon);
        const prev = best.get(k);
        if (prev !== undefined && prev <= g + 1) continue;
        // Goal test with the exit condition, checked at insertion time so a
        // trapping final state is never accepted when an exit is required.
        if (solved(nb)) {
          if (!exitRequired || nRegion.mask[entry]) {
            const pushes: Push[] = [];
            for (let t = idx; t > 0; t = sParent[t]!) pushes.push(sPush[t]!);
            pushes.reverse();
            pushes.push({ box: b, dir: di });
            return pushes;
          }
          // Solved but the player is trapped: keep searching for an exit.
          continue;
        }
        addState(nb, nRegion.canon, idx, { box: b, dir: di }, g + 1);
      }
    }
  }
  return null;
}

// Expand a push list into walk moves, starting from `player` with the given
// global box positions. Mutates boxes in place. Returns the move string or
// null if a walk is impossible.
function expandPushes(
  pushes: Push[],
  player: number,
  boxes: number[],
  wall: Uint8Array,
  w: number,
  h: number,
): { moves: string; player: number } | null {
  let moves = '';
  let p = player;
  const boxSet = new Uint8Array(w * h);
  for (const b of boxes) boxSet[b] = 1;
  for (const push of pushes) {
    const d = DIRS[push.dir]!;
    const bx = push.box % w;
    const by = (push.box / w) | 0;
    const stance = (by - d.dy) * w + (bx - d.dx);
    const dest = (by + d.dy) * w + (bx + d.dx);
    const path = walkPath(p, stance, wall, boxSet, w, h);
    if (path === null) return null;
    moves += path;
    moves += d.ch;
    // Apply the push.
    boxSet[push.box] = 0;
    boxSet[dest] = 1;
    const bi = boxes.indexOf(push.box);
    boxes[bi] = dest;
    boxes.sort((a, b) => a - b);
    p = push.box;
  }
  return { moves, player: p };
}

// Whole-level move-based A* (last resort): states are player + boxes, every
// step costs 1, heuristic is the sum of per-box nearest-goal Manhattan
// distances. Returns moves or null.
function moveAStar(level: Level, cap: number): string | null {
  const { w, h, wall, goals, goalSet } = level;
  const dist = (b: number): number => {
    let m = Infinity;
    for (const g of goals) {
      const d = Math.abs((g % w) - (b % w)) + Math.abs(((g / w) | 0) - ((b / w) | 0));
      if (d < m) m = d;
    }
    return m;
  };
  const est = (bs: number[]): number => bs.reduce((a, b) => a + dist(b), 0);
  const key = (p: number, bs: number[]): string => p + '|' + bs.join(',');
  const sp: number[] = [level.player];
  const sb: number[][] = [[...level.boxes]];
  const sg: number[] = [0];
  const parent: number[] = [-1];
  const step: string[] = [''];
  const best = new Map<string, number>([[key(level.player, level.boxes), 0]]);
  const heap: [number, number, number][] = [[est(level.boxes), 0, 0]];
  let seq = 0;
  const less = (a: [number, number, number], b: [number, number, number]): boolean =>
    a[0] !== b[0] ? a[0] < b[0] : a[1] !== b[1] ? a[1] > b[1] : a[2] < b[2];
  const push = (e: [number, number, number]): void => {
    let i = heap.length;
    heap.push(e);
    while (i > 0) {
      const pp = (i - 1) >> 1;
      if (!less(heap[i]!, heap[pp]!)) break;
      [heap[i], heap[pp]] = [heap[pp]!, heap[i]!];
      i = pp;
    }
  };
  const pop = (): [number, number, number] => {
    const top = heap[0]!;
    const last = heap.pop()!;
    if (heap.length > 0) {
      heap[0] = last;
      for (let i = 0; ; ) {
        let c = 2 * i + 1;
        if (c >= heap.length) break;
        if (c + 1 < heap.length && less(heap[c + 1]!, heap[c]!)) c++;
        if (!less(heap[c]!, heap[i]!)) break;
        [heap[i], heap[c]] = [heap[c]!, heap[i]!];
        i = c;
      }
    }
    return top;
  };
  let expanded = 0;
  while (heap.length > 0 && expanded < cap) {
    const [, g, s] = pop();
    if (best.get(key(sp[s]!, sb[s]!)) !== g) continue;
    const bs = sb[s]!;
    if (bs.every((b) => goalSet[b])) {
      let moves = '';
      for (let t = s; t > 0; t = parent[t]!) moves = step[t]! + moves;
      return moves;
    }
    expanded++;
    const p = sp[s]!;
    for (const d of DIRS) {
      const q = p + d.dy * w + d.dx;
      if (wall[q]) continue;
      let nb = bs;
      const at = bs.indexOf(q);
      if (at >= 0) {
        const dest = q + d.dy * w + d.dx;
        if (wall[dest] || bs.includes(dest)) continue;
        nb = bs.slice();
        nb[at] = dest;
        nb.sort((a, b) => a - b);
      }
      const k = key(q, nb);
      if (best.has(k) && best.get(k)! <= g + 1) continue;
      const t = sp.length;
      sp.push(q);
      sb.push(nb);
      sg.push(g + 1);
      parent.push(s);
      step.push(d.ch);
      best.set(k, t);
      push([g + 1 + est(nb), g + 1, t]);
      void seq;
    }
  }
  return null;
}

// Replay-verify moves on the level: every step legal and all boxes on goals.
function verify(level: Level, moves: string): boolean {
  const { w, wall, goalSet } = level;
  const boxSet = new Uint8Array(w * level.h);
  for (const b of level.boxes) boxSet[b] = 1;
  let p = level.player;
  for (const ch of moves.toLowerCase()) {
    const d = DIRS.find((dd) => dd.ch === ch);
    if (!d) return false;
    const q = p + d.dy * w + d.dx;
    if (wall[q]) return false;
    if (boxSet[q]) {
      const dest = q + d.dy * w + d.dx;
      if (wall[dest] || boxSet[dest]) return false;
      boxSet[q] = 0;
      boxSet[dest] = 1;
    }
    p = q;
  }
  for (let i = 0; i < boxSet.length; i++) if (boxSet[i] && !goalSet[i]) return false;
  return true;
}

function solveLevel(text: string): string {
  const level = parseLevel(text);
  const { w, h, wall, goalSet } = level;
  const hy = (level.player / w) | 0;

  // --- Decompose into rooms: connected floor components off the hallway row.
  const floor = new Uint8Array(w * h);
  for (let i = 0; i < w * h; i++) floor[i] = wall[i] ? 0 : 1;
  const seen = new Uint8Array(w * h);
  type Room = { cells: number[]; boxes: number[]; goals: number[]; doors: number[]; entries: number[] };
  const rooms: Room[] = [];
  let decomposed = true;
  for (let i = 0; i < w * h; i++) {
    const y = (i / w) | 0;
    if (!floor[i] || y === hy || seen[i]) continue;
    const cells: number[] = [];
    const q: number[] = [i];
    seen[i] = 1;
    for (let qi = 0; qi < q.length; qi++) {
      const c = q[qi]!;
      cells.push(c);
      const cx = c % w;
      const cy = (c / w) | 0;
      for (const d of DIRS) {
        const nx = cx + d.dx;
        const ny = cy + d.dy;
        if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue;
        const n = ny * w + nx;
        if (!floor[n] || (n / w | 0) === hy || seen[n]) continue;
        seen[n] = 1;
        q.push(n);
      }
    }
    const cellSet = new Uint8Array(w * h);
    for (const c of cells) cellSet[c] = 1;
    const boxes = level.boxes.filter((b) => cellSet[b]);
    const goals = level.goals.filter((g) => cellSet[g]);
    // Door cells: room cells adjacent to a hallway floor cell.
    const doors: number[] = [];
    const entries: number[] = [];
    for (const c of cells) {
      const cx = c % w;
      const cy = (c / w) | 0;
      for (const d of DIRS) {
        const nx = cx + d.dx;
        const ny = cy + d.dy;
        if (ny === hy && nx >= 0 && nx < w && floor[ny * w + nx]) {
          doors.push(c);
          entries.push(ny * w + nx);
        }
      }
    }
    doors.sort((a, b) => a - b);
    entries.sort((a, b) => a - b);
    if (boxes.length === 0 && goals.length === 0) continue; // stray floor, ignore
    if (boxes.length !== goals.length || doors.length === 0) decomposed = false;
    rooms.push({ cells, boxes, goals, doors, entries });
  }

  const boxes = [...level.boxes];
  let player = level.player;
  let allMoves = '';

  if (decomposed && rooms.length > 0) {
    // Solve each room; a room whose solution traps the player inside must
    // be solved last, so collect trapping rooms and order them last.
    type SolvedRoom = { room: Room; pushes: Push[]; traps: boolean };
    const solvedRooms: SolvedRoom[] = [];
    let ok = true;
    for (const room of rooms) {
      const cellsMask = new Uint8Array(w * h);
      for (const c of room.cells) cellsMask[c] = 1;
      const boxMask = new Uint8Array(w * h);
      for (const c of room.cells) boxMask[c] = 1;
      let found: Push[] | null = null;
      let traps = false;
      for (const entry of room.entries) {
        const spec: RoomSpec = {
          w, h, wall, goalSet,
          cells: (() => { const m = new Uint8Array(cellsMask); m[entry] = 1; return m; })(),
          boxCells: boxMask,
          boxes: room.boxes,
          goals: room.goals,
          entry,
          exitRequired: true,
        };
        found = pushSearch(spec, ROOM_ASTAR_CAP, false) ?? pushSearch(spec, ROOM_GREEDY_CAP, true);
        if (found) break;
      }
      if (!found) {
        // Retry without the exit requirement: the room may still be solvable
        // if it is handled last.
        for (const entry of room.entries) {
          const spec: RoomSpec = {
            w, h, wall, goalSet,
            cells: (() => { const m = new Uint8Array(cellsMask); m[entry] = 1; return m; })(),
            boxCells: boxMask,
            boxes: room.boxes,
            goals: room.goals,
            entry,
            exitRequired: false,
          };
          found = pushSearch(spec, ROOM_ASTAR_CAP, false) ?? pushSearch(spec, ROOM_GREEDY_CAP, true);
          if (found) break;
        }
        if (!found) {
          ok = false;
          break;
        }
        traps = true;
      }
      solvedRooms.push({ room, pushes: found, traps });
    }
    if (ok) {
      // Non-trapping rooms first (in discovery order), trapping rooms last.
      const ordered = [...solvedRooms.filter((r) => !r.traps), ...solvedRooms.filter((r) => r.traps)];
      for (const { room, pushes } of ordered) {
        // Walk to the room's entry cell, then expand its pushes.
        let entry = room.entries[0]!;
        // Pick the entry the solver actually used: it is the one whose
        // expansion succeeds; try each in order.
        let done: { moves: string; player: number } | null = null;
        for (const e of room.entries) {
          const boxSet = new Uint8Array(w * h);
          for (const b of boxes) boxSet[b] = 1;
          const toEntry = walkPath(player, e, wall, boxSet, w, h);
          if (toEntry === null) continue;
          const boxesCopy = [...boxes];
          const exp = expandPushes(pushes, e, boxesCopy, wall, w, h);
          if (exp === null) continue;
          done = { moves: toEntry + exp.moves, player: exp.player };
          for (let i = 0; i < boxes.length; i++) boxes[i] = boxesCopy[i]!;
          entry = e;
          break;
        }
        void entry;
        if (!done) {
          ok = false;
          break;
        }
        allMoves += done.moves;
        player = done.player;
      }
    }
    if (ok && verify(level, allMoves)) return allMoves;
    // Fall through to the whole-level search on any stitching failure.
    allMoves = '';
  }

  // --- Whole-level push search fallback.
  const allCells = new Uint8Array(w * h);
  for (let i = 0; i < w * h; i++) allCells[i] = wall[i] ? 0 : 1;
  const wholeSpec: RoomSpec = {
    w, h, wall, goalSet,
    cells: allCells,
    boxCells: allCells,
    boxes: [...level.boxes],
    goals: [...level.goals],
    entry: level.player,
    exitRequired: false,
  };
  const whole = pushSearch(wholeSpec, LEVEL_PUSH_CAP, false) ?? pushSearch(wholeSpec, LEVEL_PUSH_CAP, true);
  if (whole) {
    const boxesCopy = [...level.boxes];
    const exp = expandPushes(whole, level.player, boxesCopy, wall, w, h);
    if (exp && verify(level, exp.moves)) return exp.moves;
  }

  // --- Last resort: move-based A*.
  const mv = moveAStar(level, MOVE_ASTAR_CAP);
  if (mv && verify(level, mv)) return mv;
  return '';
}

for (const id of readdirSync(I).sort()) {
  const moves = solveLevel(readFileSync(join(I, id, 'level.txt'), 'utf8'));
  mkdirSync(join(O, id), { recursive: true });
  writeFileSync(join(O, id, 'solution.txt'), `${moves}\n`);
}
