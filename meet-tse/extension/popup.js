const $ = (id) => document.getElementById(id);
const DEFAULTS = { enabled: true, serverUrl: 'ws://127.0.0.1:8765/ws', disableBrowserNS: true };
const http = (ws) => ws.replace(/^ws/, 'http').replace(/\/ws$/, '');
async function refresh() {
  try {
    const s = await (await fetch(http($('url').value) + '/status')).json();
    const state = s.bypass ? 'BYPASS (model too slow; raw mic is flowing — reload the Meet tab to retry)' :
      (s.processing ? 'processing' : 'ready');
    const rtf = s.rtf == null ? '' : `\nlast measured RTF: ${Number(s.rtf).toFixed(2)}`;
    $('status').textContent = `server: online (${s.device}${s.dummy ? ', DUMMY model' : ''})\n` +
      `enrollment: ${s.enrolled ? 'ready' : 'not enrolled'}\nstate: ${state}\n` +
      `latency ≈ ${s.approx_latency_ms} ms + compute${rtf}`;
  } catch (e) { $('status').textContent = 'server: offline — start server.py'; }
}
chrome.storage.local.get(DEFAULTS, (c) => {
  $('enabled').checked = c.enabled; $('rawmic').checked = c.disableBrowserNS; $('url').value = c.serverUrl; refresh();
});
setInterval(refresh, 1000);
$('enabled').onchange = () => chrome.storage.local.set({ enabled: $('enabled').checked });
$('rawmic').onchange = () => chrome.storage.local.set({ disableBrowserNS: $('rawmic').checked });
$('url').onchange = () => { chrome.storage.local.set({ serverUrl: $('url').value }); refresh(); };
$('up').onclick = async () => {
  const f = $('file').files[0]; if (!f) return ($('status').textContent = 'choose an audio file first');
  $('status').textContent = 'uploading…';
  try {
    const r = await (await fetch(http($('url').value) + '/enroll', { method: 'POST', body: f })).json();
    $('status').textContent = r.ok ? `enrollment ready: ${r.seconds}s of audio` : `enrollment failed: ${r.msg}`;
  } catch (e) { $('status').textContent = 'upload failed: ' + e; }
  setTimeout(refresh, 1500);
};
