// A3: investigation presentation. The reducer owns exact capture identity and
// UTF-8 mapping; this view owns navigation, sample choice and DOM lifetimes.
// Model text is always inserted as text nodes. Existing source spans and sample
// articles survive polling so focus, text selection and scroll do not reset.
import { createClient, requestId } from './client.js';
import { newState, applyPage, captureAvailable, interpretationText, selectionJobs,
  generationSettings, outputSummary, selectedText, sourceSection, sourceGroups,
  findOccurrences, captureKey, investigationSamples, promptExcerpt, samplesAt, nextSampleSeed } from './controller.js';
const $ = id => document.getElementById(id);
const labels = { prompt: 'Prompt', reasoning: 'Thinking', answer: 'Answer', unclassified: 'Other' };
const client = createClient();
let state = newState(), connected = false, needsAccess = false, submitting = false;
let retryAction = null, polling = false, pollDelay = 100, capabilities = { jobs: 16, variants: 16 };
let section = 'prompt', comparing = false, lastIdentity = '', restored = false, seedInitialized = false;
let submitted = [], promptRecord = null, allSamples = [], sampleChoices = new Map();
let sourceNodes = new Map(), sourceSignature = '', pointsSignature = '', sampleSignature = '';
let matches = [], matchIndex = 0, searchSignature = '';
const phaseLabels = { idle: 'Ready for a prompt', queued: 'Queued', loading_target: 'Loading target model', generating_target: 'Generating & capturing', releasing_target: 'Retaining captures', retained: 'Captures ready', loading_verbalizer: 'Loading the NLA', interpreting: 'Interpreting', stopping: 'Stopping', unavailable: 'Run unavailable', closed: 'Service closed' };
function el(tag, text, cls) { const e = document.createElement(tag); if (text !== undefined) e.textContent = text; if (cls) e.className = cls; return e; }
function write(id, text) { if ($(id).textContent !== text) $(id).textContent = text; }
function shown(text) { return text.replace(/ /g, '␠').replace(/\n/g, '↵').replace(/\t/g, '⇥'); }
function short(text) { const label = text.trim() || shown(text) || '[empty]'; return label.length > 40 ? label.slice(0, 37) + '…' : label; }
function remember(key, value) { try { localStorage.setItem(key, JSON.stringify(value)); } catch { /* Optional metadata; journal remains authoritative. */ } }
function saved(key) { try { return JSON.parse(localStorage.getItem(key)); } catch { return null; } }
function showError(error, retry = null) { $('error').hidden = false; $('error').querySelector('span').textContent = error.message; retryAction = retry; $('retry').hidden = !retry; }
function clearError() { $('error').hidden = true; retryAction = null; }
function requireAccess(error) {
  if (!needsAccess) clearError();
  connected = false; needsAccess = true; $('access').hidden = false;
  $('access-message').textContent = error.message;
  write('connection', 'Access link needed');
  for (const id of ['generate', 'stop', 'interpret']) $(id).disabled = true;
}
function point(position) { return state.target.groups.find(g => g.first <= position && g.last >= position); }
function owner(position) {
  const group = point(position);
  if (!group || group.prompt || group.kind !== 'control') return group ? sourceSection(group) : 'prompt';
  return sourceGroups(state, 'reasoning', true).includes(group) ? 'reasoning' : sourceGroups(state, 'answer', true).includes(group) ? 'answer' : 'unclassified';
}
function capture(position) { return { ...(state.retained?.site ?? state.site), position }; }
function listAt(position) { return samplesAt(allSamples, capture(position)); }
function groupsFor(current = section) {
  if (current === 'prompt' && !$('show-controls').checked) {
    const excerpt = promptExcerpt(state, promptRecord?.prompt?.user);
    if (excerpt) return excerpt;
  }
  return sourceGroups(state, current, $('show-controls').checked);
}
function resetView() {
  section = 'prompt'; comparing = false; sampleChoices = new Map(); seedInitialized = false;
  sourceNodes = new Map(); sourceSignature = ''; pointsSignature = ''; sampleSignature = '';
  $('source').replaceChildren(); $('samples').replaceChildren(); $('sample-output').replaceChildren();
  $('compare-a').replaceChildren(); $('compare-b').replaceChildren();
  matches = []; searchSignature = ''; write('search-status', ''); $('find').value = '';
}
function select(first, last = first, locate = false) {
  if (!Number.isSafeInteger(first) || !Number.isSafeInteger(last) || first < 0 || last < first || first >= state.target.tokens.length) return;
  state.selection = [first, Math.min(last, state.target.tokens.length - 1)];
  section = owner(first); comparing = false;
  if (point(first)?.kind === 'control') $('show-controls').checked = true;
  $('position').value = String(first); $('end-position').value = String(state.selection[1]);
  render(); if (locate) locateSelection();
}
function locateSelection() {
  const target = $('source').querySelector('.selected');
  if (target) $('source').scrollTop += target.getBoundingClientRect().top - $('source').getBoundingClientRect().top - $('source').clientHeight / 2;
}
function nearbyContext(position) {
  const current = owner(position);
  const groups = current === 'prompt' ? promptExcerpt(state, promptRecord?.prompt?.user) ?? sourceGroups(state, current, false) : sourceGroups(state, current, false);
  const text = groups.filter(g => g.first <= position).map(g => g.text).join('');
  // Clip by Unicode code points rather than splitting surrogate pairs.
  return [...text].slice(-220).join('');
}
function renderSource() {
  const groups = groupsFor(), controls = $('show-controls').checked;
  const signature = `${section}/${controls}/${groups[0]?.first}`;
  if (sourceSignature !== signature) { $('source').replaceChildren(); sourceNodes.clear(); sourceSignature = signature; }
  for (const g of groups) {
    const key = `${g.first}/${g.last}`;
    let span = sourceNodes.get(key);
    if (!span) {
      span = el('span', undefined, 'token' + (g.kind === 'control' ? ' control-token' : ''));
      span.setAttribute('role', 'button'); span.tabIndex = -1;
      span.onclick = event => {
        if (event.shiftKey && state.selection) select(Math.min(state.selection[0], g.first), Math.max(state.selection[1], g.last));
        else select(g.first, g.last);
      };
      span.onkeydown = event => {
        if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); select(g.first, g.last); }
        if (!['ArrowLeft', 'ArrowRight'].includes(event.key)) return;
        event.preventDefault();
        const current = groupsFor(), index = current.indexOf(g), next = current[index + (event.key === 'ArrowRight' ? 1 : -1)];
        if (!next) return;
        if (event.shiftKey && state.selection) select(Math.min(state.selection[0], next.first), Math.max(state.selection[1], next.last));
        else select(next.first, next.last);
        sourceNodes.get(`${next.first}/${next.last}`)?.focus({ preventScroll: true }); locateSelection();
      };
      sourceNodes.set(key, span); $('source').append(span);
    }
    if (span.textContent !== g.text) span.textContent = g.text;
    const range = g.first === g.last ? String(g.first) : `${g.first}–${g.last}`;
    span.title = `${shown(g.text)} · ${labels[section]} position ${range}`;
    span.setAttribute('aria-label', span.title);
    const selected = !!state.selection && g.first <= state.selection[1] && g.last >= state.selection[0];
    span.classList.toggle('selected', selected); span.setAttribute('aria-pressed', String(selected)); span.tabIndex = -1;
  }
  const focusTarget = $('source').querySelector('.selected') ?? $('source').firstElementChild;
  if (focusTarget) focusTarget.tabIndex = 0;
  const original = section === 'prompt' && !controls && !!promptExcerpt(state, promptRecord?.prompt?.user);
  write('source-title', section === 'prompt' ? original ? 'Original prompt' : 'Model input' : section === 'reasoning' ? 'Model thinking' : section === 'answer' ? 'Answer' : 'Unclassified output');
  write('source-meta', state.selection && owner(state.selection[0]) === section ? `Selected #${state.selection.join('–').replace(/^(\d+)–\1$/, '$1')}` : 'Choose a point');
  write('source-note', section === 'prompt' ? original ? '' : 'Exact retained input, including the chat template. Positions refer to this full sequence.' : !groups.length ? section === 'answer' && state.status?.active_request && !state.retained ? 'The answer will appear after thinking finishes.' : 'No text in this section yet.' : '');
  document.querySelectorAll('[data-section]').forEach(button => {
    const key = button.dataset.section, selected = section === key;
    const count = state.target.tokens.filter(t => t.position < state.target.promptTokens ? key === 'prompt' : t.kind === key).length;
    button.setAttribute('aria-selected', String(selected)); button.tabIndex = selected ? 0 : -1;
    button.id = `tab-${key}`; button.setAttribute('aria-controls', 'source');
    button.querySelector('small').textContent = count ? count.toLocaleString() : '';
    button.hidden = key === 'unclassified' && !count;
  });
  $('source').setAttribute('aria-labelledby', `tab-${section}`);
  const laterKey = `${state.selection?.join('/')}/${state.target.tokens.length}`;
  if ($('later').dataset.key !== laterKey) {
    $('later').dataset.key = laterKey;
    const wasOpen = $('later').querySelector('details')?.open;
    $('later').replaceChildren();
    if (state.selection) {
      const p = state.selection[1], later = state.target.groups.filter(g => g.first > p && g.kind !== 'control' && !g.prompt).map(g => g.text).join('');
      if (later) { const d = el('details', undefined, 'later'); d.open = !!wasOpen; d.append(el('summary', 'Generated afterward')); d.append(el('p', 'This text was not available at the selected position.', 'hint'), el('div', later, 'later-text')); $('later').append(d); }
    }
  }
}
function renderPoints() {
  const positions = [...new Set(allSamples.map(s => s.capture.position))].sort((a, b) => a - b);
  const signature = `${positions.map(p => `${p}:${listAt(p).length}:${listAt(p).some(s => s.pending)}`).join(',')}/${state.selection?.join('/')}`;
  write('point-count', `${positions.length} points · ${allSamples.length} samples`);
  if (signature === pointsSignature) return;
  pointsSignature = signature;
  const focus = document.activeElement?.dataset.point;
  $('points').replaceChildren(); $('mobile-point').replaceChildren(el('option', 'Choose an investigated point'));
  for (const current of Object.keys(labels)) {
    const selectedPoints = positions.filter(p => owner(p) === current); if (!selectedPoints.length) continue;
    $('points').append(el('h3', labels[current]));
    for (const p of selectedPoints) {
      const samples = listAt(p), selected = state.selection?.[0] === p, button = el('button', undefined, 'point'), row = el('span', undefined, 'row');
      button.dataset.point = String(p); button.setAttribute('aria-current', String(selected));
      row.append(el('span', short(selectedText(state, p)), 'word'), el('span', String(samples.length), 'count'));
      button.append(row, el('span', `#${p} · ${samples.length} ${samples.length === 1 ? 'sample' : 'samples'}${samples.some(s => s.pending) ? ' · working' : ''}`, 'meta'));
      button.onclick = () => select(p, p, true); $('points').append(button);
      const option = el('option', `${labels[current]} · ${short(selectedText(state, p))} · #${p}`); option.value = String(p); option.selected = selected; $('mobile-point').append(option);
    }
  }
  if (!positions.length) $('points').append(el('p', 'Interpret a point to keep its samples here.', 'hint'));
  if (focus) $('points').querySelector(`[data-point="${focus}"]`)?.focus({ preventScroll: true });
}
function sampleStatus(sample) {
  if (sample.pending) return sample.stream?.generatedBytes.length ? 'Generating…' : state.status?.phase === 'loading_verbalizer' ? 'Loading the NLA…' : 'Queued…';
  if (sample.outcome === 'failed') return 'Interpretation failed';
  if (sample.outcome === 'cancelled' || sample.stop === 'user_stop') return 'Stopped · partial output';
  if (sample.stop === 'token_limit') return 'Budget reached · partial output';
  return sample.stream ? 'Finished generating' : 'No output';
}
function renderSample(container, sample) {
  if (!sample) { if (container.dataset.key !== 'empty') { container.replaceChildren(el('p', 'No interpretation at this point yet.', 'empty')); container.dataset.key = 'empty'; } return; }
  const samples = listAt(sample.capture.position), text = sample.pending && !sample.stream?.generatedBytes.length ? 'Waiting for NLA output…' : interpretationText(sample), status = sampleStatus(sample);
  if (container.dataset.key !== sample.key) {
    container.replaceChildren(); container.dataset.key = sample.key;
    const meta = el('div', undefined, 'sample-meta'); meta.append(el('span', '', 'sample-number meta'), el('span', '', 'sample-status meta'));
    const prose = el('div', undefined, 'explanation');
    const details = el('details', undefined, 'details'); details.append(el('summary', 'Generation details'), el('p', '', 'metadata'), el('pre', '', 'raw'));
    container.append(meta, prose, el('p', '', 'sample-message hint'), details);
  }
  container.querySelector('.sample-number').textContent = `Sample ${samples.findIndex(s => s.key === sample.key) + 1} of ${samples.length}`;
  container.querySelector('.sample-status').textContent = status;
  const prose = container.querySelector('.explanation');
  if (prose.dataset.text !== text) {
    prose.dataset.text = text;
    // Preserve paragraphs during streaming; update only the changed tail.
    const paragraphs = text.split(/\n\s*\n/);
    paragraphs.forEach((paragraph, i) => { let p = prose.children[i]; if (!p) { p = el('p'); prose.append(p); } if (p.textContent !== paragraph) p.textContent = paragraph; });
    while (prose.children.length > paragraphs.length) prose.lastChild.remove();
  }
  const duplicate = sample.settings?.mode === 'greedy' && samples.some(s => s.key !== sample.key && s.settings?.mode === 'greedy');
  container.querySelector('.sample-message').textContent = sample.message || (duplicate ? 'Greedy decoding repeats the same selection rule; these are not independent random samples.' : '');
  container.querySelector('.metadata').textContent = `Block ${sample.capture.block} · ${sample.capture.boundary} · position ${sample.capture.position}\nSeed ${sample.settings?.seed} · ${sample.settings?.mode} · temperature ${sample.settings?.temperature}\nStop: ${sample.stop ?? 'pending'} · format: ${sample.format === 'complete' ? 'explanation tags present' : sample.format ?? 'pending'}\nRequest ${sample.request} · job ${sample.job} · variant ${sample.variant ?? 'pending'}\nTarget ${state.retained?.pair?.target_revision ?? 'unavailable'}\nNLA ${state.retained?.pair?.verbalizer_revision ?? 'unavailable'}\nFormat does not establish accuracy.`;
  const raw = sample.stream ? new TextDecoder().decode(Uint8Array.from(sample.stream.generatedBytes)) : '';
  const pre = container.querySelector('.raw'); if (pre.textContent !== raw) pre.textContent = raw;
}
function renderSelection() {
  const selection = state.selection;
  $('selection-help').hidden = !!selection; $('selection-controls').hidden = !selection;
  if (!selection) return;
  const [first, last] = selection, count = last - first + 1, site = capture(first);
  const identityKey = `${first}/${last}/${state.retained?.site?.block ?? state.site?.block}`;
  if ($('identity').dataset.key !== identityKey) {
    $('identity').dataset.key = identityKey; $('identity').replaceChildren(el('div', 'Selected activation', 'eyebrow'));
    const line = el('div', undefined, 'selected-label'); line.append(el('strong', short(selectedText(state, first, last))), el('span', `${labels[owner(first)]} · #${first}${count > 1 ? '–' + last : ''}`, 'meta'));
    const ctx = el('p', nearbyContext(last), 'context');
    $('identity').append(line, ctx, el('p', `Context through this position · block ${site.block ?? 'pending'}${owner(first) === 'answer' ? ' · preceding thinking also belongs to this context' : ''}`, 'boundary'));
    $('token-strip').replaceChildren();
    for (let p = Math.max(0, first - 2); p <= Math.min(state.target.tokens.length - 1, last + 2, first + 10); p++) {
      const b = el('button', shown(selectedText(state, p)) || '[empty]'); b.title = `Position ${p}`; b.setAttribute('aria-label', `Position ${p}: ${shown(selectedText(state, p))}`); b.setAttribute('aria-pressed', String(p >= first && p <= last)); b.onclick = () => select(p, p, true); $('token-strip').append(b);
    }
  }
  $('previous-token').disabled = first === 0; $('next-token').disabled = first + 1 >= state.target.tokens.length;
  write('range-status', count === 1 ? 'One token · one capture' : `${count} positions · ${count} separate captures`);
  const available = Array.from({ length: count }, (_, i) => captureAvailable(state, first + i)).every(Boolean);
  const jobs = count * Number($('variant-count').value);
  const remaining = Math.max(0, capabilities.variants - state.results.filter(r => r.variant).length);
  write('capacity', `${remaining} of ${capabilities.variants} retained sample slots remaining · up to ${capabilities.jobs} jobs per request.`);
  write('token-detail', `Token ID ${state.target.tokens[first].id} · ${state.target.tokens[first].bytes.length} bytes. Each position is interpreted separately.`);
  write('capture-status', !state.retained ? 'Available to interpret when generation finishes.' : !available ? 'No completed capture for this selection.' : count > 1 ? `Interpretations below belong to position ${first}. Use the point list to read each position separately.` : '');
  $('interpret').disabled = !connected || submitting || !!retryAction || !available || !!state.status?.active_request || !!state.status?.queued_jobs || !!state.status?.publication_failed || !Number.isSafeInteger(jobs) || jobs < 1 || jobs > Math.min(capabilities.jobs, remaining);
  const samples = listAt(first);
  write('interpret', jobs > remaining ? 'Sample capacity reached' : jobs > capabilities.jobs ? 'Reduce positions or sample count' : count > 1 ? `Interpret ${count} positions` : samples.length ? 'Generate another sample' : 'Interpret this point');
  const chosen = samples.find(s => s.key === sampleChoices.get(captureKey(site))) ?? samples[0];
  const signature = samples.map(s => s.key).join(',') + '/' + chosen?.key;
  if (signature !== sampleSignature) {
    const focusKey = document.activeElement?.dataset.sample;
    sampleSignature = signature; $('samples').replaceChildren();
    samples.forEach((s, index) => { const b = el('button', String(index + 1)); b.dataset.sample = s.key; b.setAttribute('aria-label', `Sample ${index + 1}`); b.setAttribute('aria-pressed', String(s.key === chosen?.key)); b.onclick = () => { sampleChoices.set(captureKey(site), s.key); renderSelection(); $('samples').children[index]?.focus({ preventScroll: true }); }; $('samples').append(b); });
    if (focusKey) [...$('samples').children].find(b => b.dataset.sample === focusKey)?.focus({ preventScroll: true });
  }
  renderSample($('sample-output'), chosen);
  $('compare-open').disabled = allSamples.length < 2;
}
function comparisonLabel(sample) { return `${labels[owner(sample.capture.position)]} · ${short(selectedText(state, sample.capture.position))} #${sample.capture.position} · sample ${listAt(sample.capture.position).findIndex(s => s.key === sample.key) + 1}`; }
function renderCompare() {
  for (const side of ['a', 'b']) {
    const select = $('compare-' + side), value = select.value;
    if (select.options.length !== allSamples.length) {
      select.replaceChildren(...allSamples.map(s => { const option = el('option', comparisonLabel(s)); option.value = s.key; return option; }));
      if (allSamples.some(s => s.key === value)) select.value = value;
    }
    const sample = allSamples.find(s => s.key === select.value); if (!sample) continue;
    const container = $('compare-output-' + side);
    if (container.dataset.key !== sample.key) {
      container.dataset.key = sample.key; container.replaceChildren();
      const context = el('div', undefined, 'compare-context'); context.append(el('strong', `${labels[owner(sample.capture.position)]} · “${short(selectedText(state, sample.capture.position))}” · #${sample.capture.position}`), el('span', nearbyContext(sample.capture.position), 'meta'));
      container.append(context, el('div', undefined, 'comparison-sample'));
    }
    renderSample(container.querySelector('.comparison-sample'), sample);
  }
  const a = allSamples.find(s => s.key === $('compare-a').value), b = allSamples.find(s => s.key === $('compare-b').value);
  write('comparison-title', a?.key === b?.key ? 'Same sample selected twice' : a && b && captureKey(a.capture) === captureKey(b.capture) ? 'Same activation, different samples' : 'Two points in the same generation');
}
function render() {
  const identity = `${state.epoch}/${state.run}`;
  if (lastIdentity !== identity) { lastIdentity = identity; resetView(); }
  allSamples = investigationSamples(state, submitted);
  if (!seedInitialized && state.status && state.cursor === state.status.latest) {
    $('av-seed').value = nextSampleSeed(allSamples); seedInitialized = true;
  }
  const active = !!state.status?.active_request || !!state.status?.queued_jobs;
  write('connection', connected ? phaseLabels[state.status?.phase] ?? 'Working' : needsAccess ? 'Access link needed' : 'Disconnected — reconnecting');
  $('generate').disabled = !connected || active || submitting || !!retryAction;
  write('generate', state.run ? 'Generate again' : 'Generate');
  $('stop').disabled = !connected || !state.status?.active_request;
  $('stop').hidden = !state.status?.active_request;
  write('stop', state.retained ? 'Stop interpretation' : 'Stop generation');
  $('replace-note').hidden = !state.run;
  write('run-label', state.run ? `Experiment ${state.run}` : 'New experiment');
  write('experiment-prompt', promptRecord?.prompt?.user ?? (state.run ? 'Explore the retained generation.' : 'Generate, select a point, then explore its activation.'));
  $('reader').hidden = comparing; $('inspector').hidden = comparing; $('compare').hidden = !comparing;
  const summary = outputSummary(state);
  write('target-status', state.run ? `${summary.generated.toLocaleString()} generated tokens${state.retained ? ' · captures ready' : ''}` : 'Enter a prompt above to begin.');
  let notice = '';
  if (summary.stop === 'token_limit') notice = summary.answer ? 'Output budget reached. The answer may be incomplete.' : 'Output budget reached before an answer. Captured thinking is still available to explore.';
  if (summary.stop === 'user_stop') notice = 'Generation stopped. Completed captures remain available.';
  if (summary.stop === 'execution_failure') notice = 'Generation failed. Any completed captures remain available.';
  if (state.status?.phase === 'unavailable') notice = 'This run is unavailable. Start a new experiment.';
  $('generation-notice').hidden = !notice; write('generation-notice', notice);
  const working = allSamples.filter(s => s.pending), elsewhere = working.filter(s => s.capture.position !== state.selection?.[0]);
  const unknown = active && state.retained && !working.length;
  $('background-status').hidden = !elsewhere.length && !unknown;
  write('background-status', elsewhere.length ? `${elsewhere.length} sample jobs in progress at ${[...new Set(elsewhere.map(s => '#' + s.capture.position))].join(', ')}. Your selection stays here.` : unknown ? 'An interpretation is running from another browser. Its point will appear when the service publishes the result.' : '');
  renderPoints(); if (comparing) renderCompare(); else { renderSource(); renderSelection(); }
  if (state.status?.publication_failed) showError(new Error('Live publication failed. This transcript may be incomplete; start a new experiment after the operation stops.'));
}
function settings(prefix) {
  return generationSettings({ mode: $(prefix + '-mode').value, seed: $(prefix + '-seed').value, max_new_tokens: Number($(prefix + '-cap').value), temperature: Number($(prefix + '-temperature').value), top_p: prefix === 'av' ? 1 : Number($('target-top-p').value) });
}
function restoreMetadata() {
  submitted = saved('dolittle-jobs') ?? [];
  if (!Array.isArray(submitted)) submitted = [];
  submitted = submitted.filter(r => r.epoch === state.epoch && r.run === state.run && Array.isArray(r.jobs));
  promptRecord = saved('dolittle-prompt');
  if (!promptRecord) { try { promptRecord = JSON.parse(sessionStorage.getItem('dolittle-prompt')); } catch { /* Legacy metadata is optional. */ } }
  if (promptRecord?.epoch !== state.epoch || promptRecord?.run !== state.run) promptRecord = null;
  if (promptRecord) {
    $('prompt').value = promptRecord.prompt.user; $('system').value = promptRecord.prompt.system ?? '';
    $('thinking').checked = promptRecord.prompt.thinking === 'enabled';
    for (const [name, key] of Object.entries({ cap: 'max_new_tokens', mode: 'mode', seed: 'seed', temperature: 'temperature', 'top-p': 'top_p' })) $('target-' + name).value = promptRecord.settings[key];
    remember('dolittle-prompt', promptRecord); write('prompt-preview', promptRecord.prompt.user);
    $('target-temperature').disabled = promptRecord.settings.mode === 'greedy';
    $('target-top-p').disabled = promptRecord.settings.mode === 'greedy';
  } else write('prompt-preview', state.run ? 'Retained experiment · open to start another prompt' : 'Start with a question or a small puzzle.');
}
async function command(path, body, requestEpoch = state.epoch) {
  const retry = () => command(path, body, requestEpoch);
  try {
    clearError(); submitting = true; render();
    const receipt = await client.command(path, body, requestEpoch);
    if (body.prompt) {
      promptRecord = { epoch: requestEpoch, run: receipt.run, prompt: body.prompt, settings: body.settings };
      remember('dolittle-prompt', promptRecord); $('composer').open = false; write('prompt-preview', body.prompt.user);
    }
    if (body.jobs?.length && $('av-seed').value === body.jobs[0].settings.seed) {
      const next = BigInt(body.jobs[0].settings.seed) + BigInt(body.jobs.length);
      if (next <= 18446744073709551615n) $('av-seed').value = next.toString();
    }
    if (receipt.run !== state.run) {
      state = newState(state.epoch, receipt.run); submitted = []; remember('dolittle-jobs', submitted); resetView();
      lastIdentity = `${state.epoch}/${state.run}`;
      if (body.prompt) section = body.prompt.thinking === 'enabled' ? 'reasoning' : 'answer';
    }
    await poll();
  } catch (error) {
    if (error.status) {
      submitted = submitted.filter(r => r.request !== body.request); remember('dolittle-jobs', submitted);
      if (error.status === 401) requireAccess(error); else showError(error);
    } else showError(error, retry);
  } finally { submitting = false; render(); }
}
async function poll() {
  if (polling) return;
  polling = true;
  try {
    const status = await client.status(); connected = true; needsAccess = false;
    if (!$('access').hidden) clearError(); $('access').hidden = true;
    if (status.epoch !== state.epoch || status.run !== state.run) {
      if (state.epoch && state.run) showError(new Error('The service restarted or the experiment changed. Previous captures are no longer available.'));
      state = newState(status.epoch, status.run); restored = false;
    }
    if (!restored) { restored = true; restoreMetadata(); if (status.run) $('composer').open = false; }
    capabilities = status.limits; state.status = status; write('pair', status.pair);
    if (state.run) {
      const page = await client.events(state.run, state.cursor); applyPage(state, page);
      pollDelay = page.more || state.status.active_request || state.status.queued_jobs ? 100 : 1000;
    } else pollDelay = 1000;
    render();
  } catch (error) {
    connected = false; pollDelay = error.status === 401 ? 1000 : Math.min(Math.max(pollDelay * 2, 1000), 10000);
    if (error.status === 401) requireAccess(error);
    else { needsAccess = false; write('connection', 'Disconnected — reconnecting'); for (const id of ['generate', 'stop', 'interpret']) $(id).disabled = true; }
  } finally { polling = false; }
}
for (const prefix of ['target', 'av']) $(prefix + '-mode').onchange = () => {
  const greedy = $(prefix + '-mode').value === 'greedy'; $(prefix + '-temperature').disabled = greedy;
  if (prefix === 'target') $('target-top-p').disabled = greedy;
};
$('prompt-form').addEventListener('invalid', event => { for (let d = event.target.closest('details'); d; d = d.parentElement.closest('details')) d.open = true; }, true);
$('prompt-form').onsubmit = event => {
  event.preventDefault();
  try { const system = $('system').value; command('/api/v1/runs', { request: requestId(), replace: state.run, prompt: { user: $('prompt').value, thinking: $('thinking').checked ? 'enabled' : 'disabled', ...(system ? { system } : {}) }, settings: settings('target') }); } catch (error) { showError(error); }
};
$('stop').onclick = () => { if (state.status?.active_request) command(`/api/v1/runs/${state.run}/stop`, { request: requestId(), active_request: state.status.active_request }); };
$('interpret').onclick = () => {
  try {
    const jobs = selectionJobs(state, state.selection[0], state.selection[1], Number($('variant-count').value), settings('av'), capabilities.jobs), request = requestId();
    submitted.push({ epoch: state.epoch, run: state.run, request, jobs }); remember('dolittle-jobs', submitted);
    // The initiating point selects its new slot once; later polling never steals selection.
    for (let index = 0; index < jobs.length; index++) if (!jobs.slice(0, index).some(j => captureKey(j.capture) === captureKey(jobs[index].capture))) sampleChoices.set(captureKey(jobs[index].capture), `${request}/${index}`);
    command(`/api/v1/runs/${state.run}/interpretations`, { request, jobs });
  } catch (error) { showError(error); }
};
$('previous-token').onclick = () => select(state.selection[0] - 1, state.selection[0] - 1, true);
$('next-token').onclick = () => select(state.selection[0] + 1, state.selection[0] + 1, true);
$('position').onchange = () => select(Number($('position').value), Number($('position').value), true);
$('end-position').onchange = () => select(Number($('position').value), Number($('end-position').value), true);
$('variant-count').oninput = renderSelection;
$('av-seed').oninput = () => { seedInitialized = true; };
$('show-controls').onchange = renderSource;
$('mobile-point').onchange = event => { if (/^\d+$/.test(event.target.value)) select(Number(event.target.value), Number(event.target.value), true); };
$('back-text').onclick = () => { section = owner(state.selection[0]); renderSource(); $('reader').scrollIntoView({ block: 'start' }); locateSelection(); $('source').querySelector('.selected')?.focus({ preventScroll: true }); };
$('retry').onclick = () => retryAction?.();
$('access-form').onsubmit = async event => { event.preventDefault(); try { client.connect($('access-link').value); $('access-link').value = ''; clearError(); write('access-message', 'Connecting…'); await poll(); } catch (error) { showError(error); } };
$('composer').ontoggle = () => write('composer-action', $('composer').open ? 'Collapse' : state.run ? 'New generation' : 'Edit prompt');
const tabs = [...document.querySelectorAll('[data-section]')];
function switchSection(button) { section = button.dataset.section; matches = []; searchSignature = ''; write('search-status', ''); renderSource(); }
for (const button of tabs) {
  button.onclick = () => switchSection(button);
  button.onkeydown = event => { if (!['ArrowLeft', 'ArrowRight'].includes(event.key)) return; event.preventDefault(); const visible = tabs.filter(b => !b.hidden), next = visible[(visible.indexOf(button) + (event.key === 'ArrowRight' ? 1 : visible.length - 1)) % visible.length]; switchSection(next); next.focus(); };
}
function search(delta = 0) {
  const signature = `${section}/${$('show-controls').checked}/${$('find').value}/${state.target.tokens.length}`;
  if (signature !== searchSignature || !delta) { matches = findOccurrences(groupsFor(), $('find').value); matchIndex = 0; searchSignature = signature; }
  else if (matches.length) matchIndex = (matchIndex + delta + matches.length) % matches.length;
  if (!matches.length) { write('search-status', 'No match'); return; }
  const [first, last] = matches[matchIndex]; select(first, last, true); write('search-status', `${matchIndex + 1}/${matches.length}`);
}
$('search-form').onsubmit = event => { event.preventDefault(); search(); };
$('next-match').onclick = () => search(1); $('previous-match').onclick = () => search(-1);
$('compare-open').onclick = () => {
  comparing = true; render();
  const list = listAt(state.selection[0]), a = list.find(s => s.key === sampleChoices.get(captureKey(capture(state.selection[0])))) ?? list[0] ?? allSamples[0], b = list.find(s => s.key !== a.key) ?? allSamples.find(s => s.key !== a.key);
  $('compare-a').value = a.key; $('compare-b').value = b.key; renderCompare(); $('compare-close').focus();
};
$('compare-close').onclick = () => { comparing = false; render(); $('compare-open').focus({ preventScroll: true }); };
$('compare-a').onchange = renderCompare; $('compare-b').onchange = renderCompare;
async function loop() { await poll(); setTimeout(loop, pollDelay); }
render(); loop();
