const messagesEl = document.getElementById('messages');
const inputEl = document.getElementById('input');
const sendBtn = document.getElementById('send');
const attachLabel = document.getElementById('attach');
const fileInput = document.getElementById('file');
const sessionsEl = document.getElementById('sessions');
const toggleBtn = document.getElementById('toggle-sessions');

let messages = [];
let attachments = [];
let sessions = ['Session 1', 'Session 2'];
let currentSession = 'Session 1';
let isSessionExpanded = true;

function renderSessions() {
  sessionsEl.innerHTML = '';
  sessions.forEach((s) => {
    const li = document.createElement('li');
    li.className = 'session-item' + (s === currentSession ? ' active' : '');
    li.textContent = s;
    li.addEventListener('click', () => {
      currentSession = s;
      renderSessions();
    });
    sessionsEl.appendChild(li);
  });
}

function renderMessages() {
  messagesEl.innerHTML = '';
  messages.forEach((m) => {
    const wrapper = document.createElement('div');
    wrapper.className = 'message-row ' + (m.role === 'user' ? 'right' : 'left');

    const bubble = document.createElement('div');
    bubble.className = 'message-bubble ' + (m.role === 'user' ? 'user' : 'assistant');
    bubble.textContent = m.content;
    wrapper.appendChild(bubble);

    if (m.attachments && m.attachments.length) {
      const att = document.createElement('div');
      att.className = 'attachments';
      m.attachments.forEach((a) => {
        const aEl = document.createElement('a');
        aEl.href = '#';
        aEl.textContent = a;
        aEl.className = 'attachment-link';
        att.appendChild(aEl);
      });
      wrapper.appendChild(att);
    }

    messagesEl.appendChild(wrapper);
  });
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

async function sendMessage() {
  const text = inputEl.value.trim();
  if (!text) return;
  const newMessages = [...messages, { role: 'user', content: text, attachments: attachments.map(f => f.name) }];
  messages = newMessages;
  inputEl.value = '';
  attachments = [];
  renderMessages();

  try {
    const res = await fetch('/api/chat', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ messages: newMessages }) });
    const data = await res.json();
    if (data.response) {
      messages = [...newMessages, { role: 'assistant', content: data.response }];
    } else if (data.result && typeof data.result === 'string') {
      messages = [...newMessages, { role: 'assistant', content: data.result }];
    } else {
      messages = [...newMessages, { role: 'assistant', content: 'Sorry, could not process the request.' }];
    }
    renderMessages();
  } catch (e) {
    messages = [...newMessages, { role: 'assistant', content: 'Error contacting server.' }];
    renderMessages();
  }
}

sendBtn.addEventListener('click', sendMessage);
inputEl.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
});

attachLabel.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', (e) => {
  const files = Array.from(e.target.files || []);
  attachments.push(...files);
});

toggleBtn.addEventListener('click', () => {
  isSessionExpanded = !isSessionExpanded;
  document.getElementById('sidebar').classList.toggle('collapsed', !isSessionExpanded);
  toggleBtn.textContent = isSessionExpanded ? 'Shrink' : 'Expand';
});

renderSessions();
renderMessages();
