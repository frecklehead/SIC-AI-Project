// Isolated-world content script: relays PCM between the page (inject.js) and
// the MV3 service worker, which owns the WebSocket to the local TSE server.
// The socket cannot live here: recent Chrome blocks content-script connections
// to 127.0.0.1 with ERR_BLOCKED_BY_LOCAL_NETWORK_ACCESS_CHECKS.
const DEFAULTS = { enabled: true, serverUrl: 'ws://127.0.0.1:8765/ws', prebufferMs: 150, disableBrowserNS: true };
const conns = new Map(); // sid -> {port, alive, sampleRate}

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
const toPage = (msg, transfer) => window.postMessage({ src: 'tse-bridge', ...msg }, '*', transfer || []);
async function pushConfig() {
  const { enabled, prebufferMs, disableBrowserNS } = await getCfg();
  toPage({ type: 'config', cfg: { enabled, prebufferMs, disableBrowserNS } });
}

function connect(sid, sampleRate) {
  const old = conns.get(sid);
  if (old) { old.alive = false; try { old.port.disconnect(); } catch (_) {} }
  const port = chrome.runtime.connect({ name: 'tse' });
  const c = { port, alive: true, sampleRate };
  conns.set(sid, c);
  port.onMessage.addListener((m) => {
    if (!m || m.sid !== sid) return;
    if (m.type === 'state') toPage({ type: 'state', sid, ready: m.ready, msg: m.msg || '' });
    else if (m.type === 'pcm') { const buf = b64decode(m.b64); toPage({ type: 'pcm', sid, buf }, [buf]); }
  });
  port.onDisconnect.addListener(() => {
    if (!c.alive) return;
    toPage({ type: 'state', sid, ready: false, msg: 'server unavailable' });
    setTimeout(() => { if (c.alive) connect(sid, sampleRate); }, 1000);  // worker restart
  });
  port.postMessage({ type: 'start', sid, sampleRate });
}

window.addEventListener('message', (e) => {
  const m = e.data;
  if (e.source !== window || !m || m.src !== 'tse-page') return;
  if (m.type === 'hello') pushConfig();
  else if (m.type === 'start') connect(m.sid, m.sampleRate);
  else if (m.type === 'pcm') {
    const c = conns.get(m.sid);
    if (c && c.alive) {
      try { c.port.postMessage({ type: 'pcm', sid: m.sid, b64: b64encode(m.buf) }); }
      catch (_) { c.alive = false; }
    }
  } else if (m.type === 'stop') {
    const c = conns.get(m.sid);
    if (c) { c.alive = false; try { c.port.disconnect(); } catch (_) {} conns.delete(m.sid); }
  }
});
chrome.storage.onChanged.addListener(pushConfig);
pushConfig();
