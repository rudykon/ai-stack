const state = {
  models: { chat: [], image: [] },
  messages: [],
  activeView: 'chat',
  stack: null
};

const $ = (id) => document.getElementById(id);

function setHealth(text) {
  $('healthText').textContent = text;
}

function option(label, value, selected = false) {
  const item = document.createElement('option');
  item.textContent = label;
  item.value = value;
  item.selected = selected;
  return item;
}

function fillSelect(select, models) {
  select.innerHTML = '';
  models.forEach((model, index) => {
    select.appendChild(option(model.label || model.id, model.id, Boolean(model.default) || index === 0));
  });
}

function selectedModel(kind) {
  const select = kind === 'chat' ? $('chatModel') : $('imageModel');
  return state.models[kind].find((model) => model.id === select.value) || state.models[kind][0];
}

function applyChatDefaults() {
  const model = selectedModel('chat');
  const defaults = model?.defaults || {};
  if (defaults.temperature !== undefined) $('temperature').value = defaults.temperature;
  if (defaults.max_tokens !== undefined) $('maxTokens').value = defaults.max_tokens;
  $('enableThinking').checked = Boolean(defaults.enable_thinking);
}

function applyImageDefaults() {
  const model = selectedModel('image');
  const defaults = model?.defaults || {};
  const sizes = model?.sizes?.length ? model.sizes : ['256x256', '512x512', '768x768', '1024x1024'];
  $('imageSize').innerHTML = '';
  sizes.forEach((size) => $('imageSize').appendChild(option(size, size, size === defaults.size)));
  if (defaults.steps !== undefined) $('steps').value = defaults.steps;
  if (defaults.true_cfg_scale !== undefined) $('cfgScale').value = defaults.true_cfg_scale;
}

async function getJson(path) {
  const response = await fetch(path);
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    const detail = typeof data.detail === 'string' ? data.detail : JSON.stringify(data.detail || data);
    throw new Error(detail || `HTTP ${response.status}`);
  }
  return data;
}

async function api(path, body) {
  const response = await fetch(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body)
  });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    const detail = typeof data.detail === 'string' ? data.detail : JSON.stringify(data.detail || data);
    throw new Error(detail || `HTTP ${response.status}`);
  }
  return data;
}

function renderMessages() {
  const box = $('chatMessages');
  box.innerHTML = '';
  if (!state.messages.length) {
    const empty = document.createElement('div');
    empty.className = 'empty-state';
    empty.textContent = 'No messages yet';
    box.appendChild(empty);
    return;
  }
  state.messages.forEach((message) => {
    const item = document.createElement('div');
    item.className = `message ${message.role}`;
    const role = document.createElement('div');
    role.className = 'message-role';
    role.textContent = message.role;
    const body = document.createElement('div');
    body.className = 'message-body';
    body.textContent = message.content;
    item.append(role, body);
    box.appendChild(item);
  });
  box.scrollTop = box.scrollHeight;
}

function addMessage(role, content) {
  state.messages.push({ role, content });
  renderMessages();
}

function showError(content) {
  const item = document.createElement('div');
  item.className = 'message error';
  const role = document.createElement('div');
  role.className = 'message-role';
  role.textContent = 'error';
  const body = document.createElement('div');
  body.className = 'message-body';
  body.textContent = content;
  item.append(role, body);
  $('chatMessages').appendChild(item);
}

async function sendChat(event) {
  event.preventDefault();
  const input = $('chatInput');
  const text = input.value.trim();
  if (!text) return;

  addMessage('user', text);
  input.value = '';
  $('sendChat').disabled = true;
  setHealth('Chat running');

  try {
    const data = await api('/api/chat', {
      model: $('chatModel').value,
      messages: state.messages,
      temperature: Number($('temperature').value),
      max_tokens: Number($('maxTokens').value),
      enable_thinking: $('enableThinking').checked
    });
    addMessage('assistant', data.content || '');
    setHealth('Ready');
  } catch (error) {
    showError(error.message);
    setHealth('Error');
  } finally {
    $('sendChat').disabled = false;
    input.focus();
  }
}

function renderImages(items) {
  const box = $('imageResults');
  box.innerHTML = '';
  if (!items.length) {
    const empty = document.createElement('div');
    empty.className = 'empty-state';
    empty.textContent = 'No image returned';
    box.appendChild(empty);
    return;
  }

  items.forEach((item, index) => {
    const card = document.createElement('div');
    card.className = 'image-item';
    const img = document.createElement('img');
    img.alt = `Generated image ${index + 1}`;
    img.src = item.b64_json ? `data:image/png;base64,${item.b64_json}` : item.url;
    const meta = document.createElement('div');
    meta.className = 'image-meta';
    const left = document.createElement('span');
    left.textContent = `${item.width || ''}x${item.height || ''}`;
    const right = document.createElement('span');
    right.textContent = item.seed === null || item.seed === undefined ? 'auto seed' : `seed ${item.seed}`;
    meta.append(left, right);
    card.append(img, meta);
    box.appendChild(card);
  });
}

async function generateImage(event) {
  event.preventDefault();
  const prompt = $('imagePrompt').value.trim();
  if (!prompt) return;

  $('generateImage').disabled = true;
  setHealth('Image running');
  const placeholder = document.createElement('div');
  placeholder.className = 'empty-state';
  placeholder.textContent = 'Generating';
  $('imageResults').replaceChildren(placeholder);

  try {
    const seedValue = $('seed').value.trim();
    const data = await api('/api/images', {
      model: $('imageModel').value,
      prompt,
      negative_prompt: $('negativePrompt').value || ' ',
      size: $('imageSize').value,
      steps: Number($('steps').value),
      true_cfg_scale: Number($('cfgScale').value),
      seed: seedValue ? Number(seedValue) : null,
      n: 1,
      add_magic_prompt: $('magicPrompt').checked
    });
    renderImages(data.data || []);
    setHealth('Ready');
  } catch (error) {
    const item = document.createElement('div');
    item.className = 'empty-state';
    item.textContent = error.message;
    $('imageResults').replaceChildren(item);
    setHealth('Error');
  } finally {
    $('generateImage').disabled = false;
  }
}

function statusLabel(item) {
  if (item.status !== 'ready') return 'offline';
  if (item.worker_running === false) return 'idle';
  if (item.loaded === false) return 'ready';
  return 'ready';
}

function statusClass(status) {
  return status === 'ready' ? 'ready' : status === 'idle' ? 'idle' : 'offline';
}

function makeStatusPill(status) {
  const pill = document.createElement('span');
  pill.className = `status-pill ${statusClass(status)}`;
  pill.textContent = status;
  return pill;
}

function renderLayerCards(layers) {
  const box = $('layerCards');
  box.innerHTML = '';
  layers.forEach((layer) => {
    const card = document.createElement('div');
    card.className = 'layer-card';
    const top = document.createElement('div');
    top.className = 'layer-top';
    const name = document.createElement('h3');
    name.textContent = layer.name;
    top.append(name, makeStatusPill(layer.status));
    const primary = document.createElement('div');
    primary.className = 'layer-primary';
    primary.textContent = layer.primary || '';
    const detail = document.createElement('p');
    detail.textContent = layer.detail || '';
    card.append(top, primary, detail);
    box.appendChild(card);
  });
}

function renderModelHealth(items) {
  const box = $('modelHealth');
  box.innerHTML = '';
  items.forEach((item) => {
    const row = document.createElement('div');
    row.className = 'health-row';
    const main = document.createElement('div');
    const name = document.createElement('strong');
    name.textContent = item.label || item.id;
    const meta = document.createElement('span');
    meta.textContent = `${item.type || 'model'} · ${item.runner || 'runner'}`;
    main.append(name, meta);
    const status = statusLabel(item);
    row.append(main, makeStatusPill(status));

    const endpoint = document.createElement('code');
    endpoint.textContent = item.base_url || '';
    const wrap = document.createElement('div');
    wrap.className = 'health-item';
    wrap.append(row, endpoint);
    if (item.detail) {
      const detail = document.createElement('p');
      detail.textContent = item.detail;
      wrap.appendChild(detail);
    }
    box.appendChild(wrap);
  });
}

async function copyText(text) {
  if (!navigator.clipboard) return;
  await navigator.clipboard.writeText(text);
  setHealth('Copied');
  setTimeout(() => setHealth('Ready'), 900);
}

function renderSnippets(snippets) {
  const names = {
    open_webui: 'Open WebUI',
    litellm_yaml: 'LiteLLM YAML',
    chat_curl: 'Chat curl',
    image_curl: 'Image curl'
  };
  const box = $('integrationSnippets');
  box.innerHTML = '';
  Object.entries(snippets || {}).forEach(([key, value]) => {
    const item = document.createElement('div');
    item.className = 'snippet-item';
    const heading = document.createElement('div');
    heading.className = 'snippet-heading';
    const title = document.createElement('strong');
    title.textContent = names[key] || key;
    const button = document.createElement('button');
    button.className = 'ghost-button small-button';
    button.type = 'button';
    button.textContent = 'Copy';
    button.addEventListener('click', () => copyText(value));
    heading.append(title, button);
    const pre = document.createElement('pre');
    pre.textContent = value;
    item.append(heading, pre);
    box.appendChild(item);
  });
}

function renderReferences(items) {
  const box = $('referenceList');
  box.innerHTML = '';
  if (!items.length) {
    const empty = document.createElement('p');
    empty.textContent = 'No references registered';
    box.appendChild(empty);
    return;
  }
  items.forEach((item) => {
    const row = document.createElement('div');
    row.className = 'reference-row';
    const name = document.createElement('strong');
    name.textContent = item.name || item.id || 'reference';
    const role = document.createElement('span');
    role.textContent = item.role || '';
    const file = document.createElement('code');
    file.textContent = item.local || item.url || '';
    row.append(name, role, file);
    box.appendChild(row);
  });
}

function renderStack(data) {
  state.stack = data;
  $('gatewayText').textContent = `${data.gateway_url} · API key ${data.api_key}`;
  renderLayerCards(data.layers || []);
  renderModelHealth(data.health || []);
  renderSnippets(data.integrations || {});
  renderReferences(data.references || []);
}

async function loadStack() {
  try {
    const data = await getJson('/api/stack');
    renderStack(data);
  } catch (error) {
    $('gatewayText').textContent = error.message;
  }
}

function switchView(view) {
  state.activeView = view;
  $('chatTab').classList.toggle('active', view === 'chat');
  $('imageTab').classList.toggle('active', view === 'image');
  $('stackTab').classList.toggle('active', view === 'stack');
  $('chatView').classList.toggle('active', view === 'chat');
  $('imageView').classList.toggle('active', view === 'image');
  $('stackView').classList.toggle('active', view === 'stack');
  if (view === 'stack') loadStack();
}

async function loadModels() {
  const data = await getJson('/api/models');
  state.models.chat = data.chat || [];
  state.models.image = data.image || [];
  fillSelect($('chatModel'), state.models.chat);
  fillSelect($('imageModel'), state.models.image);
  applyChatDefaults();
  applyImageDefaults();
  setHealth('Ready');
}

function clearActive() {
  if (state.activeView === 'chat') {
    state.messages = [];
    renderMessages();
  } else if (state.activeView === 'image') {
    $('imagePrompt').value = '';
    $('negativePrompt').value = '';
    $('imageResults').innerHTML = '';
  } else {
    loadStack();
  }
}

function bindEvents() {
  $('chatForm').addEventListener('submit', sendChat);
  $('imageForm').addEventListener('submit', generateImage);
  $('chatTab').addEventListener('click', () => switchView('chat'));
  $('imageTab').addEventListener('click', () => switchView('image'));
  $('stackTab').addEventListener('click', () => switchView('stack'));
  $('clearButton').addEventListener('click', clearActive);
  $('refreshStack').addEventListener('click', loadStack);
  $('chatModel').addEventListener('change', applyChatDefaults);
  $('imageModel').addEventListener('change', applyImageDefaults);
}

bindEvents();
renderMessages();
loadModels()
  .then(loadStack)
  .catch((error) => setHealth(error.message));
