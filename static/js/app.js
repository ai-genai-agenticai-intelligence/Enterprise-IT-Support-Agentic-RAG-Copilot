const chat = document.getElementById('chat');
const form = document.getElementById('chatForm');
const question = document.getElementById('question');
const trace = document.getElementById('trace');
const sourceUsed = document.getElementById('sourceUsed');

const backdrop = document.getElementById('backdrop');
const sidebar = document.getElementById('sidebar');
const tracePanel = document.getElementById('tracePanel');
const toggleSidebarBtn = document.getElementById('toggleSidebar');
const closeSidebarBtn = document.getElementById('closeSidebar');
const toggleTraceBtn = document.getElementById('toggleTrace');
const closeTraceBtn = document.getElementById('closeTrace');
const modal = document.getElementById('uploadModal');
const openUploadBtn = document.getElementById('openUpload');
const mobileUploadBtn = document.getElementById('mobileUploadBtn');
const closeUploadBtn = document.getElementById('closeUpload');

function closeDrawers() {
  if (sidebar) sidebar.classList.remove('open');
  if (tracePanel) tracePanel.classList.remove('open');
  if (backdrop) backdrop.classList.add('hidden');
}

function openSidebar() {
  if (tracePanel) tracePanel.classList.remove('open');
  if (sidebar) sidebar.classList.add('open');
  if (backdrop) backdrop.classList.remove('hidden');
}

function openTrace() {
  if (sidebar) sidebar.classList.remove('open');
  if (tracePanel) tracePanel.classList.add('open');
  if (backdrop) backdrop.classList.remove('hidden');
}

if (toggleSidebarBtn) toggleSidebarBtn.onclick = openSidebar;
if (closeSidebarBtn) closeSidebarBtn.onclick = closeDrawers;
if (toggleTraceBtn) toggleTraceBtn.onclick = openTrace;
if (closeTraceBtn) closeTraceBtn.onclick = closeDrawers;
if (backdrop) backdrop.onclick = closeDrawers;

if (openUploadBtn) openUploadBtn.onclick = () => { closeDrawers(); modal.classList.remove('hidden'); };
if (mobileUploadBtn) mobileUploadBtn.onclick = () => { closeDrawers(); modal.classList.remove('hidden'); };
if (closeUploadBtn) closeUploadBtn.onclick = () => modal.classList.add('hidden');

function escapeHtml(s = '') { return s.replace(/[&<>'"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#39;', '"': '&quot;' }[c])); }
function formatText(s = '') { return escapeHtml(s).replace(/\n/g, '<br>'); }

function addMessage(role, text, source = '', citations = []) {
  const wrap = document.createElement('div'); wrap.className = `message ${role}`;
  const citeHtml = citations.length ? `<div class="citations"><strong>Sources</strong><br>${citations.map(c => c.url ? `<a href="${escapeHtml(c.url)}" target="_blank" rel="noopener">${escapeHtml(c.title)}</a>` : escapeHtml(c.title)).join('<br>')}</div>` : '';
  wrap.innerHTML = `<div class="avatar">${role === 'user' ? 'You' : 'AI'}</div><div class="bubble">${formatText(text)}${source ? `<div class="answer-source">Source: ${escapeHtml(source)}</div>` : ''}${citeHtml}</div>`;
  chat.appendChild(wrap);
  chat.scrollTop = chat.scrollHeight;
}

function renderTrace(items = []) {
  trace.innerHTML = items.length ? items.map(x => `<div class="trace-item">${escapeHtml(x)}</div>`).join('') : '<div class="empty">No trace.</div>';
}

async function askAgent(q) {
  closeDrawers();
  addMessage('user', q);
  question.value = '';
  renderTrace(['Running LangGraph workflow...']);
  sourceUsed.textContent = 'Running';
  const btn = form.querySelector('button');
  btn.disabled = true;
  try {
    const res = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question: q })
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'Request failed');
    addMessage('assistant', data.answer, data.source_used, data.citations || []);
    renderTrace(data.trace || []);
    sourceUsed.textContent = data.source_used;
  } catch (e) {
    addMessage('assistant', `Error: ${e.message}`);
    renderTrace(['Request failed']);
    sourceUsed.textContent = 'Error';
  } finally {
    btn.disabled = false;
    chat.scrollTop = chat.scrollHeight;
  }
}

form.addEventListener('submit', e => {
  e.preventDefault();
  const q = question.value.trim();
  if (q) askAgent(q);
});

question.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey && window.innerWidth > 768) {
    e.preventDefault();
    form.dispatchEvent(new Event('submit', { cancelable: true }));
  }
});

document.querySelectorAll('.example').forEach(b => b.addEventListener('click', () => askAgent(b.textContent.trim())));

document.getElementById('uploadBtn').onclick = async () => {
  const file = document.getElementById('fileInput').files[0];
  const key = document.getElementById('adminKey').value;
  const status = document.getElementById('uploadStatus');
  if (!file) { status.textContent = 'Choose a file first.'; return; }
  status.textContent = 'Indexing document...';
  const fd = new FormData();
  fd.append('file', file);
  try {
    const r = await fetch('/api/ingest', { method: 'POST', headers: { 'X-Admin-Key': key }, body: fd });
    const d = await r.json();
    if (!r.ok) throw new Error(d.detail || 'Upload failed');
    status.textContent = `Indexed ${d.file}: ${d.chunks} chunks.`;
  } catch (e) {
    status.textContent = `Error: ${e.message}`;
  }
};