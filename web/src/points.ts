/**
 * 予測の点の後処理と、正解との突き合わせ。scripts/eval_points.py と同じ決まりにする。
 * - 切片の重なりで同じ物に 2 つ付いた予測は、確信度の高い方に寄せる（距離 25px、配布元の nms_WSI.py）
 * - 正解の順に、7.5µm 以内でいちばん近い、まだ使っていない予測を当てる（evalutils の score_detection）
 */

export interface Pred {
  x: number;
  y: number;
  score: number;
  cls: number;
}

export interface Truth {
  x: number;
  y: number;
  cls: number;
}

export const NMS_PX = 25;
export const RADIUS_UM = 7.5;

/** 区分ごとに、確信度の高い順で、近いものを消す */
export function nms(preds: Pred[], r = NMS_PX): Pred[] {
  const out: Pred[] = [];
  const sorted = [...preds].sort((a, b) => b.score - a.score);
  // 格子に入れて近所だけを見る
  const grid = new Map<string, Pred[]>();
  const key = (x: number, y: number) => `${Math.floor(x / r)},${Math.floor(y / r)}`;
  for (const p of sorted) {
    const gx = Math.floor(p.x / r);
    const gy = Math.floor(p.y / r);
    let near = false;
    for (let dy = -1; dy <= 1 && !near; dy++) {
      for (let dx = -1; dx <= 1 && !near; dx++) {
        for (const q of grid.get(`${gx + dx},${gy + dy}`) ?? []) {
          if (q.cls === p.cls && Math.hypot(q.x - p.x, q.y - p.y) <= r) {
            near = true;
            break;
          }
        }
      }
    }
    if (near) continue;
    out.push(p);
    const k = key(p.x, p.y);
    grid.set(k, [...(grid.get(k) ?? []), p]);
  }
  return out;
}

export type Mark = "tp" | "fp" | "fn";

export interface Score {
  tp: number;
  fp: number;
  fn: number;
  /** 予測ごとの判定（preds と同じ順） */
  pred: Mark[];
  /** 正解ごとの判定（truth と同じ順。tp か fn） */
  truth: Mark[];
}

/** 分裂像（区分 0）だけで突き合わせる */
export function score(truth: Truth[], preds: Pred[], rPx: number): Score {
  const g = truth.map((t, i) => ({ ...t, i })).filter((t) => t.cls === 0);
  const used = preds.map(() => false);
  const tMark: Mark[] = truth.map(() => "fn");
  let tp = 0;
  let fn = 0;
  for (const t of g) {
    let best = -1;
    let bestD = Infinity;
    preds.forEach((p, j) => {
      if (used[j]) return;
      const d = Math.hypot(p.x - t.x, p.y - t.y);
      if (d <= rPx && d < bestD) {
        bestD = d;
        best = j;
      }
    });
    if (best >= 0) {
      used[best] = true;
      tp++;
      tMark[t.i] = "tp";
    } else fn++;
  }
  return { tp, fp: used.filter((u) => !u).length, fn, pred: used.map((u) => (u ? "tp" : "fp")), truth: tMark };
}
