// MV3 service worker: owns the WebSocket connections to the local TSE server.
//
// Recent Chrome applies the Local Network Access check to content scripts running
// with the page's origin (meet.google.com), which blocks ws://127.0.0.1 with
// ERR_BLOCKED_BY_LOCAL_NETWORK_ACCESS_CHECKS.  Extension-origin contexts are
// allowed (host_permissions include 127.0.0.1), so the socket lives here and
// bridge.js only relays PCM over a runtime port.
const DEFAULTS = { serverUrl: 'ws://127.0.0.1:8765/ws' };
const conns = new Map(); // sid -> {ws, ready, alive, retry, post, sampleRate}

// chrome.runtime messaging is JSON-only, so PCM crosses the port as base64.
function b64decode(b64) {
  const s = atob(b64);
  const u8 = new Uint8Array(s.length);
  for (let i = 0; i < s.length; i++) u8[i] = s.charCodeAt(i);
  return u8.buffer;
}
function b64encode(buf) {
  const u8 = new Uint8Array(buf);
  let s = '';
  for (let i = 0; i < u8.length; i += 0x8000)
    s += String.fromCharCode.apply(null, u8.subarray(i, i + 0x8000));
  return btoa(s);
}

const getCfg = () => new Promise((r) => chrome.storage.local.get(DEFAULTS, r));

async function open(sid) {
  const c = conns.get(sid);
  if (!c || !c.alive) return;
  clearTimeout(c.retry);
  const gen = (c.gen = (c.gen || 0) + 1);   // invalidate handlers of an older socket
  if (c.ws) { try { c.ws.onclose = null; c.ws.close(); } catch (_) {} }
  const { serverUrl } = await getCfg();
  if (!c.alive || c.gen !== gen) return;
  let ws;
  try { ws = new WebSocket(serverUrl); }
  catch (e) {
    c.post({ type: 'state', sid, ready: false, msg: String(e) });
    c.retry = setTimeout(() => open(sid), 2000);
    return;
  }
  ws.binaryType = 'arraybuffer';
  c.ws = ws; c.ready = false;
  ws.onopen = () => ws.send(JSON.stringify({ type: 'start', sampleRate: c.sampleRate }));
  ws.onmessage = (ev) => {
    if (!c.alive || c.gen !== gen) return;
    if (typeof ev.data === 'string') {
      let m = {};
      try { m = JSON.parse(ev.data); } catch (_) {}
      c.ready = m.type === 'ready';
      c.post({ type: 'state', sid, ready: c.ready, msg: m.msg || '' });
    } else {
      c.post({ type: 'pcm', sid, b64: b64encode(ev.data) });
    }
  };
  ws.onclose = () => {
    if (!c.alive || c.gen !== gen) return;
    c.ready = false;
    c.post({ type: 'state', sid, ready: false, msg: 'server unavailable' });
    c.retry = setTimeout(() => open(sid), 2000);
  };
  ws.onerror = () => {};
}

function close(sid) {
  const c = conns.get(sid);
  if (!c) return;
  c.alive = false; c.ready = false;
  c.gen = (c.gen || 0) + 1;
  clearTimeout(c.retry);
  try { c.ws && c.ws.close(); } catch (_) {}
  conns.delete(sid);
}

chrome.runtime.onConnect.addListener((port) => {
  if (port.name !== 'tse') return;
  const owned = new Set();
  const post = (m) => { try { port.postMessage(m); } catch (_) {} };
  port.onMessage.addListener((m) => {
    if (!m || m.sid == null) return;
    if (m.type === 'start') {
      owned.add(m.sid);
      const old = conns.get(m.sid);
      if (old) close(m.sid);
      conns.set(m.sid, { ws: null, ready: false, alive: true, retry: null, post, sampleRate: m.sampleRate });
      open(m.sid);
    } else if (m.type === 'pcm') {
      const c = conns.get(m.sid);
      if (c && c.ready && c.ws && c.ws.readyState === 1) {
        try { c.ws.send(b64decode(m.b64)); } catch (_) {}   // onclose reconnects
      }
    } else if (m.type === 'stop') {
      owned.delete(m.sid);
      close(m.sid);
    }
  });
  port.onDisconnect.addListener(() => { for (const sid of owned) close(sid); });
});
