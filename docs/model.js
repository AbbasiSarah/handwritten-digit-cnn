// DigitCNN inference in plain JavaScript (same maths as src/cnn2d.py).
const DigitNet = (() => {
  // x: Float32Array of C*H*W ; W: rows = C*9 (c, ki, kj), cols = Cout
  function conv3(x, C, H, Wd, W, b) {
    const O = b.length, out = new Float32Array(O * H * Wd);
    for (let o = 0; o < O; o++) {
      for (let y = 0; y < H; y++) for (let xx = 0; xx < Wd; xx++) {
        let s = b[o];
        for (let c = 0; c < C; c++) for (let i = 0; i < 3; i++) {
          const yy = y + i - 1;
          if (yy < 0 || yy >= H) continue;
          for (let j = 0; j < 3; j++) {
            const xj = xx + j - 1;
            if (xj < 0 || xj >= Wd) continue;
            s += x[(c * H + yy) * Wd + xj] * W[c * 9 + i * 3 + j][o];
          }
        }
        out[(o * H + y) * Wd + xx] = s > 0 ? s : 0;
      }
    }
    return out;
  }
  function pool2(x, C, H, Wd) {
    const h = H / 2, w = Wd / 2, out = new Float32Array(C * h * w);
    for (let c = 0; c < C; c++) for (let y = 0; y < h; y++) for (let xx = 0; xx < w; xx++) {
      let m = -Infinity;
      for (let i = 0; i < 2; i++) for (let j = 0; j < 2; j++) m = Math.max(m, x[(c * H + 2 * y + i) * Wd + 2 * xx + j]);
      out[(c * h + y) * w + xx] = m;
    }
    return out;
  }
  function dense(x, W, b, relu) {
    const out = new Float32Array(b.length);
    for (let o = 0; o < b.length; o++) {
      let s = b[o];
      for (let i = 0; i < x.length; i++) s += x[i] * W[i][o];
      out[o] = relu && s < 0 ? 0 : s;
    }
    return out;
  }
  // img: Float32Array(784), values 0..1 (white digit on black, like MNIST)
  function predict(img, P) {
    let x = conv3(img, 1, 28, 28, P['0_W'], P['0_b']);
    x = pool2(x, 8, 28, 28);
    x = conv3(x, 8, 14, 14, P['3_W'], P['3_b']);
    x = pool2(x, 16, 14, 14);
    x = dense(x, P['7_W'], P['7_b'], true);
    const z = dense(x, P['10_W'], P['10_b'], false);
    const m = Math.max(...z), e = Array.from(z, v => Math.exp(v - m)), s = e.reduce((a, v) => a + v, 0);
    return e.map(v => v / s);
  }
  return { predict };
})();
if (typeof module !== 'undefined') module.exports = DigitNet;
