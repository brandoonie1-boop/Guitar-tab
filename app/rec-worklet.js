// AudioWorklet that hands recorded audio to the page in blocks (stereo Float32), plus a level.
class GtRecorder extends AudioWorkletProcessor {
  constructor() {
    super();
    this.size = 16384;
    this.l = new Float32Array(this.size);
    this.r = new Float32Array(this.size);
    this.n = 0;
    this.peak = 0;
    this.on = true;
    this.port.onmessage = (e) => {
      if (e.data === 'flush') this.flush();
      else if (e.data === 'stop') { this.flush(); this.on = false; }
    };
  }

  flush() {
    if (!this.n) return;
    const l = this.l.slice(0, this.n), r = this.r.slice(0, this.n);
    this.port.postMessage({ l, r, peak: this.peak }, [l.buffer, r.buffer]);
    this.n = 0;
    this.peak = 0;
  }

  process(inputs) {
    if (!this.on) return false;
    const inp = inputs[0];
    if (inp && inp.length) {
      const a = inp[0], b = inp[1] || inp[0];
      for (let i = 0; i < a.length; i++) {
        this.l[this.n] = a[i];
        this.r[this.n] = b[i];
        const v = Math.abs(a[i]) > Math.abs(b[i]) ? Math.abs(a[i]) : Math.abs(b[i]);
        if (v > this.peak) this.peak = v;
        this.n++;
        if (this.n === this.size) this.flush();
      }
    }
    return true;
  }
}
registerProcessor('gt-recorder', GtRecorder);
