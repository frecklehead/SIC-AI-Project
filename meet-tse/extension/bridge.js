// Isolated-world content script: owns the websockets to the local TSE server
// (isolated worlds are not bound by meet.google.com's CSP) and relays audio to/from inject.js.
const DEFAULTS = { enabled: true, serverUrl: 'ws://127.0.0.1:8765/ws', prebufferMs: 150, disableBrowserNS: true };
const conns = new Map(); // sid -> {ws, sampleRate, alive, retry}

const getCfg = () => new Promise((r) => chrome.storage.local.get(DEFAULTS, r));
const toPage = (msg, transfer) => window.postMessage({ src: 'tse-bridge', ...msg }, '*', transfer || []);
async function pushConfig() {
  const { enabled, prebufferMs, disableBrowserNS } = await getCfg();
  toPage({ type: 'config', cfg: { enabled, prebufferMs, disableBrowserNS } });
}

async function open(sid) {
  const c = conns.get(sid); if (!c || !c.alive) return;
  const { serverUrl } = await getCfg();
  let ws;
  try { ws = new WebSocket(serverUrl); } catch (e) { toPage({ type: 'state', sid, ready: false, msg: String(e) }); return; }
  ws.binaryType = 'arraybuffer'; c.ws = ws; c.ready = false;
  ws.onopen = () => ws.send(JSON.stringify({ type: 'start', sampleRate: c.sampleRate }));
  ws.onmessage = (ev) => {
    if (typeof ev.data === 'string') {
      const m = JSON.parse(ev.data);
      c.ready = m.type === 'ready';
      toPage({ type: 'state', sid, ready: c.ready, msg: m.msg || '' });
    } else toPage({ type: 'pcm', sid, buf: ev.data }, [ev.data]);
  };
  const lost = () => {
    c.ready = false; toPage({ type: 'state', sid, ready: false, msg: 'server unavailable' });
    if (c.alive) c.retry = setTimeout(() => open(sid), 2000);
  };
  ws.onclose = lost; ws.onerror = () => {};
}

window.addEventListener('message', (e) => {
  const m = e.data;
  if (e.source !== window || !m || m.src !== 'tse-page') return;
  if (m.type === 'hello') pushConfig();
  else if (m.type === 'start') { conns.set(m.sid, { sampleRate: m.sampleRate, alive: true }); open(m.sid); }
  else if (m.type === 'pcm') { const c = conns.get(m.sid); if (c && c.ready && c.ws.readyState === 1) c.ws.send(m.buf); }
  else if (m.type === 'stop') { const c = conns.get(m.sid); if (c) { c.alive = false; clearTimeout(c.retry); try { c.ws && c.ws.close(); } catch (_) {} conns.delete(m.sid); } }
});
chrome.storage.onChanged.addListener(pushConfig);
pushConfig();
