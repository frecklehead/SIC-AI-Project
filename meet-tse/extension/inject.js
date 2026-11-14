// Runs in the PAGE (MAIN world) before Meet's own scripts.
// Wraps getUserMedia: the real mic track is piped through
//   MediaStreamTrackProcessor -> [local TSE server via bridge.js] -> MediaStreamTrackGenerator
// and the generated track is what Meet (and therefore WebRTC) receives.
// Fail-open: if the server is unreachable/not enrolled, audio passes through untouched.
(() => {
  if (window.__tseInstalled) return;
  window.__tseInstalled = true;

  const cfg = { enabled: true, prebufferMs: 150, maxBufferMs: 500, disableBrowserNS: true };
  const sessions = new Map(); // sid -> {ready}
  // Unique per tab: the worker keys connections by sid, so two Meet tabs with
  // sid=1 would otherwise close each other's socket.
  let nextSid = 1 + Math.floor(Math.random() * 1e9);

  window.addEventListener('message', (e) => {
    const m = e.data;
    if (e.source !== window || !m || m.src !== 'tse-bridge') return;
    if (m.type === 'config') Object.assign(cfg, m.cfg);
    else if (m.type === 'state') { const s = sessions.get(m.sid); if (s) { s.ready = m.ready; s.msg = m.msg || ''; } }
    else if (m.type === 'pcm') { const s = sessions.get(m.sid); if (s) s.onPcm(new Int16Array(m.buf)); }
  });
  const post = (msg, transfer) => window.postMessage({ src: 'tse-page', ...msg }, '*', transfer || []);
  post({ type: 'hello' });

  class Ring {
    constructor(cap) { this.b = new Float32Array(cap); this.r = 0; this.len = 0; }
    write(a) {
      const cap = this.b.length;
      if (a.length >= cap) { a = a.subarray(a.length - cap); this.r = 0; this.len = 0; }
      const over = this.len + a.length - cap;
      if (over > 0) { this.r = (this.r + over) % cap; this.len -= over; }
      let w = (this.r + this.len) % cap;
      for (let i = 0; i < a.length; i++) { this.b[w] = a[i]; w = (w + 1) % cap; }
      this.len += a.length;
    }
    read(out) {
      const cap = this.b.length;
      for (let i = 0; i < out.length; i++) { out[i] = this.b[this.r]; this.r = (this.r + 1) % cap; }
      this.len -= out.length;
    }
    clear() { this.r = 0; this.len = 0; }
  }

  function makeProcessedTrack(inTrack) {
    const sid = nextSid++;
    const sess = { ready: false, msg: '' };
    sessions.set(sid, sess);
    let started = false, sr = 48000, ring = null, playing = false, stats = { under: 0 };
    const processor = new MediaStreamTrackProcessor({ track: inTrack });
    const generator = new MediaStreamTrackGenerator({ kind: 'audio' });

    sess.onPcm = (i16) => {
      if (!ring) return;
      const f = new Float32Array(i16.length);
      for (let i = 0; i < i16.length; i++) f[i] = i16[i] / 32768;
      ring.write(f);
    };

    const ts = new TransformStream({
      transform(frame, controller) {
        const n = frame.numberOfFrames, ch = frame.numberOfChannels;
        if (!started) {
          sr = frame.sampleRate; started = true;
          ring = new Ring(Math.ceil(sr * cfg.maxBufferMs / 1000));
          post({ type: 'start', sid, sampleRate: sr });
        }
        const mono = new Float32Array(n);
        frame.copyTo(mono, { planeIndex: 0, format: 'f32-planar' });
        if (ch > 1) {
          const t = new Float32Array(n);
          for (let c = 1; c < ch; c++) { frame.copyTo(t, { planeIndex: c, format: 'f32-planar' }); for (let i = 0; i < n; i++) mono[i] += t[i]; }
          for (let i = 0; i < n; i++) mono[i] /= ch;
        }
        const timestamp = frame.timestamp;
        frame.close();

        let out = mono;
        if (cfg.enabled && sess.ready && generator.enabled) {
          const i16 = new Int16Array(n);
          for (let i = 0; i < n; i++) { const v = Math.max(-1, Math.min(1, mono[i])); i16[i] = v < 0 ? v * 32768 : v * 32767; }
          post({ type: 'pcm', sid, buf: i16.buffer }, [i16.buffer]);
          if (!playing && ring.len >= sr * cfg.prebufferMs / 1000) playing = true;
          if (playing && ring.len >= n) { out = new Float32Array(n); ring.read(out); }
          else if (playing) { playing = false; stats.under++; }   // underrun -> raw mic, re-buffer
          // else: still pre-buffering -> keep the raw mic, never silence
        } else {
          playing = false; if (ring) ring.clear();
        }
        controller.enqueue(new AudioData({ format: 'f32-planar', sampleRate: sr, numberOfFrames: n,
                                           numberOfChannels: 1, timestamp, data: out }));
      },
      flush() { post({ type: 'stop', sid }); sessions.delete(sid); },
    });
    processor.readable.pipeThrough(ts).pipeTo(generator.writable).catch(() => {});

    const origStop = generator.stop.bind(generator);
    generator.stop = () => { try { inTrack.stop(); } catch (_) {} post({ type: 'stop', sid }); sessions.delete(sid); origStop(); };
    inTrack.addEventListener('ended', () => { post({ type: 'stop', sid }); });
    return generator;
  }

  const md = navigator.mediaDevices;
  const origGUM = md.getUserMedia.bind(md);
  md.getUserMedia = async function (constraints) {
    let c = constraints;
    if (c && c.audio && cfg.disableBrowserNS) {
      const a = typeof c.audio === 'object' ? c.audio : {};
      c = { ...c, audio: { ...a, noiseSuppression: false, autoGainControl: false } };
    }
    const stream = await origGUM(c);
    try {
      const a = stream.getAudioTracks()[0];
      if (!a || !constraints || !constraints.audio) return stream;
      if (typeof MediaStreamTrackProcessor === 'undefined' || typeof MediaStreamTrackGenerator === 'undefined') {
        console.warn('[TSE] Insertable Streams not available in this browser; passing mic through.');
        return stream;
      }
      const out = new MediaStream([...stream.getVideoTracks(), makeProcessedTrack(a)]);
      console.log('[TSE] mic wrapped');
      return out;
    } catch (err) {
      console.error('[TSE] failed, passing mic through', err);
      return stream;
    }
  };
})();
