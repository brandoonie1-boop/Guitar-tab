// A fretboard under the tab that lights up the notes as they play (the recording or the tab),
// like a teacher's hand on the neck. Notes come from the tab player's events (app/guitarsynth.js
// scoreEvents), on the same timeline as the cursor.

const NS = 'http://www.w3.org/2000/svg';

export class LiveFretboard {
  constructor(el) {
    this.el = el;
    this.events = [];
    this.lit = new Set();
    this.i = 0;
  }

  /** events: [{t, end, s (0 = low E), fret}], capo: frets from the nut */
  setPart(events, capo = 0) {
    this.events = events.slice().sort((a, b) => a.t - b.t);
    this.capo = capo;
    const top = Math.max(12, ...this.events.map((e) => e.fret + capo));
    this.frets = Math.min(24, top + 1);
    this._build();
    this.i = 0;
  }

  _build() {
    const n = this.frets, W = 900, H = 120, x0 = 34, x1 = W - 8, y0 = 14, y1 = H - 22;
    // frets get narrower up the neck, as on a guitar
    const scale = (k) => 1 - Math.pow(2, -k / 12);
    const fx = (k) => x0 + ((x1 - x0) * scale(k)) / scale(n);
    const sy = (s) => y1 - ((y1 - y0) * s) / 5; // high string on top, as in the tab
    this.fx = fx; this.sy = sy;
    const svg = document.createElementNS(NS, 'svg');
    svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
    svg.setAttribute('class', 'live-neck');
    svg.setAttribute('role', 'img');
    svg.setAttribute('aria-label', 'Fretboard showing the notes as they play');
    const add = (tag, attrs, text) => {
      const e = document.createElementNS(NS, tag);
      for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
      if (text != null) e.textContent = text;
      svg.append(e);
      return e;
    };
    add('rect', { x: x0, y: y0 - 4, width: x1 - x0, height: y1 - y0 + 8, rx: 3, class: 'wood' });
    add('rect', { x: x0 - 5, y: y0 - 6, width: 5, height: y1 - y0 + 12, class: 'nut' });
    for (let k = 1; k <= n; k++) add('line', { x1: fx(k), y1: y0 - 4, x2: fx(k), y2: y1 + 4, class: 'fret' });
    const mid = (k) => (fx(k - 1) + fx(k)) / 2;
    for (const k of [3, 5, 7, 9, 15, 17, 19, 21]) if (k <= n) add('circle', { cx: mid(k), cy: (y0 + y1) / 2, r: 4, class: 'inlay' });
    for (const k of [12, 24]) if (k <= n) { add('circle', { cx: mid(k), cy: (y0 + y1) / 2 - 18, r: 4, class: 'inlay' }); add('circle', { cx: mid(k), cy: (y0 + y1) / 2 + 18, r: 4, class: 'inlay' }); }
    for (let s = 0; s < 6; s++) add('line', { x1: x0, y1: sy(s), x2: x1, y2: sy(s), class: 'string', 'stroke-width': 0.8 + (5 - s) * 0.35 });
    for (const k of [3, 5, 7, 9, 12, 15, 17, 19, 21, 24]) if (k <= n) add('text', { x: mid(k), y: H - 4, class: 'fretno', 'text-anchor': 'middle' }, String(k));
    if (this.capo) add('rect', { x: fx(this.capo) - 11, y: y0 - 9, width: 9, height: y1 - y0 + 18, rx: 3, class: 'capo' });
    // a dot for every place a note can sit; lit while its note sounds
    this.dots = new Map();
    for (let s = 0; s < 6; s++) {
      for (let k = 0; k <= n; k++) {
        const cx = k === 0 ? x0 - 18 : mid(k);
        const g = document.createElementNS(NS, 'g');
        g.setAttribute('class', 'dot');
        const c = document.createElementNS(NS, 'circle');
        c.setAttribute('cx', cx); c.setAttribute('cy', sy(s)); c.setAttribute('r', 9);
        const t = document.createElementNS(NS, 'text');
        t.setAttribute('x', cx); t.setAttribute('y', sy(s) + 4); t.setAttribute('text-anchor', 'middle');
        g.append(c, t);
        svg.append(g);
        this.dots.set(`${s}:${k}`, { g, t });
      }
    }
    this.el.replaceChildren(svg);
    this.lit.clear();
  }

  /** Light the notes sounding at time t (seconds on the recording's timeline). */
  update(t) {
    const ev = this.events;
    if (!ev.length || !this.dots) return;
    // events start in order: walk from the last position, or search after a jump
    if (this.i >= ev.length || (this.i > 0 && ev[this.i - 1].t > t)) {
      let lo = 0, hi = ev.length;
      while (lo < hi) { const m = (lo + hi) >> 1; if (ev[m].t <= t) lo = m + 1; else hi = m; }
      this.i = lo;
    }
    while (this.i < ev.length && ev[this.i].t <= t) this.i++;
    const now = new Set();
    // notes that started up to 4 s ago and still sound (short ones stay lit a moment)
    for (let j = this.i - 1; j >= 0 && ev[j].t > t - 4; j--) {
      const e = ev[j];
      if (t < Math.max(e.end, e.t + 0.15)) now.add(e);
    }
    const keys = new Map();
    for (const e of now) keys.set(`${e.s}:${e.fret + (this.capo || 0)}`, e);
    for (const k of this.lit) if (!keys.has(k)) { const d = this.dots.get(k); if (d) d.g.classList.remove('on'); }
    for (const [k, e] of keys) {
      const d = this.dots.get(k);
      if (!d) continue;
      d.t.textContent = String(e.fret);
      d.g.classList.add('on');
    }
    this.lit = new Set(keys.keys());
  }
}
