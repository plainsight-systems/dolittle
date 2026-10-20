// A3: DOM presentation. The controller owns identity/state; model output is
// always assigned through textContent. Nothing here interprets model control IDs.
import { createClient, requestId } from './client.js';
import { newState, applyPage, captureAvailable, interpretationText, selectionJobs, generationSettings, outputSummary, selectedText } from './controller.js';
const $ = id => document.getElementById(id);
const client = createClient();
let state = newState(), connected = false, retryAction = null, pollDelay = 100;
let capabilities = { jobs: 16 }, needsAccess = false;
let rendered = new Map(), resultCount = 0, lastRun = null, submitting = false;
let pendingInterpretation = null, restored = false;
const phaseLabels = { idle: 'Ready for a prompt', queued: 'Queued', loading_target: 'Loading target model', generating_target: 'Generating & capturing', releasing_target: 'Retaining captures', retained: 'Ready to explore', loading_verbalizer: 'Loading activation verbalizer', interpreting: 'Interpreting selection', stopping: 'Stopping after capture finalization', unavailable: 'Run unavailable — start a new experiment', closed: 'Service closed' };
function showError(error, retry = null) { $('error').hidden = false; $('error').querySelector('span').textContent = error.message; retryAction = retry; $('retry').hidden = !retry; }
function clearError() { $('error').hidden = true; retryAction = null; }
function resetView() { rendered = new Map(); resultCount = 0; for (const id of ['prompt-tokens', 'reasoning', 'answer', 'unclassified', 'interpretations']) $(id).replaceChildren(); for (const id of ['prompt-section', 'reasoning-section', 'answer-section', 'unclassified-section']) $(id).hidden = true; }
function select(first, last = first) {
  if (!Number.isSafeInteger(first) || !Number.isSafeInteger(last) || first < 0 || last < first || first >= state.target.tokens.length) return;
  state.selection = [first, Math.min(last, state.target.tokens.length - 1)];
  $('position').value = String(first); $('end-position').value = String(state.selection[1]); renderSelection();
}
function renderSelection() {
  const selection = state.selection;
  $('selection-help').hidden = !!selection; $('selection-controls').hidden = !selection;
  $('selection-jump').hidden = !selection;
  for (const [index, button] of rendered) {
    const group = state.target.groups[index];
    const selected = !!selection && group.first <= selection[1] && group.last >= selection[0];
    button.classList.toggle('selected', selected);
    button.setAttribute('aria-pressed', String(selected));
    button.tabIndex = -1;
  }
  const focusEntry = [...rendered].filter(([, button]) => !button.closest('details:not([open])')).find(([index]) => !state.target.groups[index].prompt && state.target.groups[index].kind !== 'control' && (selection ? state.target.groups[index].last >= selection[0] : state.target.groups[index].text.trim()));
  if (focusEntry) focusEntry[1].tabIndex = 0;
  if (!selection) return;
  const [first, last] = selection, token = state.target.tokens[first];
  $('selection-text').textContent = selectedText(state, first, last) || '[empty token]';
  $('token-detail').textContent = `Token ID ${token.id} · ${token.bytes.length} raw bytes. Each position is interpreted separately.`;
  const site = state.retained?.site ?? state.site;
  const count = last - first + 1, jobs = count * Number($('variant-count').value);
  const available = Array.from({ length: count }, (_, i) => captureAvailable(state, first + i)).every(Boolean);
  $('capture-status').textContent = state.retained
    ? available ? `Position ${first}${count > 1 ? '–' + last : ''} · block ${site.block} · ${count} ${count === 1 ? 'token' : 'tokens'}` : 'No completed capture for this selection.'
    : 'You can interpret this selection when generation finishes.';
  $('interpret').disabled = !connected || submitting || !available || jobs > capabilities.jobs || jobs < 1 || !Number.isSafeInteger(jobs) || !!state.status?.active_request || !!state.status?.queued_jobs || !!state.status?.publication_failed;
  $('interpret').textContent = jobs > capabilities.jobs ? `Select up to ${capabilities.jobs} token / variant pairs` : count === 1 ? 'Interpret this token' : `Interpret ${count} tokens`;
  $('previous').disabled = first === 0; $('next').disabled = first + 1 >= state.target.tokens.length;
}
function renderOutputState() {
  const summary = outputSummary(state), hasRun = !!state.run;
  const busy = !state.retained && !!state.status?.active_request;
  $('empty-target').hidden = hasRun;
  $('answer-section').hidden = !hasRun;
  $('answer-placeholder').hidden = !hasRun || !!summary.answer;
  $('answer-placeholder').textContent = busy ? 'The answer will appear after thinking finishes.' : 'No answer was generated.';
  if (!summary.thinking && busy) $('answer-placeholder').textContent = 'Waiting for model output…';
  $('reasoning-status').textContent = `${summary.thinking} tokens · ${summary.answer ? 'Finished' : busy ? 'In progress' : 'Stopped'}`;
  $('answer-link').hidden = !summary.answer;
  $('answer-status').textContent = summary.answer ? busy ? 'In progress' : summary.stop === 'stop_token' ? 'Complete' : 'Partial' : '';
  $('target-status').textContent = hasRun ? `${summary.generated} generated tokens${busy ? ' · ' + (phaseLabels[state.status?.phase] ?? 'Working') : state.retained ? ' · Captures ready' : ''}` : 'Your output will appear here.';
  let notice = '';
  if (summary.stop === 'token_limit') notice = summary.answer
    ? 'Output budget reached. The answer may be incomplete. Increase the budget and generate again to allow more room.'
    : 'Output budget reached before an answer was generated. Increase the budget and generate again. You can still explore the captured thinking.';
  else if (summary.stop === 'user_stop') notice = 'Generation stopped. Completed captures are available to explore.';
  else if (summary.stop === 'execution_failure') notice = 'Generation failed. Any completed captures remain available.';
  $('generation-notice').hidden = !notice; $('generation-notice').textContent = notice;
  $('selection-guide').hidden = !summary.generated;
}
function renderLiveInterpretation() {
  const pending = pendingInterpretation && !state.requests.has(pendingInterpretation.request);
  const phase = state.status?.phase;
  const active = phase === 'loading_verbalizer' || phase === 'interpreting';
  const streams = [...state.attempts].filter(([, stream]) => !stream.finished);
  const current = streams.at(-1)?.[1];
  $('live-interpretation').hidden = !pending && !active;
  $('live-title').textContent = phase === 'loading_verbalizer' ? 'Loading the NLA…' : current ? 'NLA output · generating' : 'Interpretation queued…';
  $('live-output').textContent = current ? new TextDecoder().decode(Uint8Array.from(current.generatedBytes)) || 'Reading the selected activation…' : pending ? `Selected text: “${pendingInterpretation.text}”` : 'Reading the selected activation…';
  if (pendingInterpretation && state.requests.has(pendingInterpretation.request)) pendingInterpretation = null;
}

function render() {
  if (lastRun !== state.run) { resetView(); $('reasoning-section').open = true; lastRun = state.run; }
  const active = !!state.status?.active_request || !!state.status?.queued_jobs;
  $('connection').textContent = connected ? (phaseLabels[state.status?.phase] ?? 'Connecting…') : 'Disconnected — reconnecting';
  $('generate').disabled = !connected || active || submitting || !!retryAction;
  $('generate').textContent = state.run ? 'Generate again' : 'Generate';
  $('stop').disabled = !connected || !state.status?.active_request;
  $('stop').hidden = !state.status?.active_request;
  $('stop').textContent = state.status?.phase === 'interpreting' || state.status?.phase === 'loading_verbalizer' ? 'Stop interpretation' : 'Stop generation';
  $('replace-note').hidden = !state.run;
  $('run-label').textContent = state.run ? `Experiment ${state.run}` : 'No active experiment';
  const controls = $('show-controls').checked;
  state.target.groups.forEach((group, index) => {
    const control = group.prompt || group.kind === 'control';
    if (control && !controls) return;
    const section = control ? 'prompt' : group.kind === 'reasoning' ? 'reasoning' : group.kind === 'answer' ? 'answer' : 'unclassified';
    const container = $(section === 'prompt' ? 'prompt-tokens' : section); $(section + '-section').hidden = false;
    let button = rendered.get(index);
    if (!button) { button = document.createElement('button'); button.type = 'button'; button.className = 'token' + (control ? ' control-token' : ''); button.addEventListener('click', event => {
      if (event.shiftKey && state.selection) select(Math.min(state.selection[0], group.first), Math.max(state.selection[1], group.last));
      else select(group.first, group.last);
    });
    button.addEventListener('keydown', event => {
      if (!['ArrowLeft', 'ArrowRight'].includes(event.key)) return;
      event.preventDefault();
      const visible = [...rendered].filter(([, button]) => !button.closest('details:not([open])')).map(([key]) => key), nextIndex = visible[visible.indexOf(index) + (event.key === 'ArrowRight' ? 1 : -1)];
      if (nextIndex === undefined) return;
      const next = state.target.groups[nextIndex];
      if (event.shiftKey && state.selection) select(Math.min(state.selection[0], next.first), Math.max(state.selection[1], next.last));
      else select(next.first, next.last);
      rendered.get(nextIndex).focus();
    }); container.append(button); rendered.set(index, button); }
    button.textContent = group.text; button.title = `Position ${group.first}${group.last !== group.first ? '–' + group.last : ''}`;
  });
  while (resultCount < state.results.length) {
    const result = state.results[resultCount++];
    const article = document.createElement('article'); article.className = 'interpretation';
    const title = document.createElement('h3'); title.textContent = `Position ${result.capture.position} · interpretation ${result.variant ?? 'unavailable'}`;
    const excerpt = document.createElement('blockquote'); excerpt.textContent = selectedText(state, result.capture.position) || '[empty token]';
    const text = document.createElement('p'); text.textContent = interpretationText(result);
    const status = document.createElement('p'); status.className = 'hint';
    status.textContent = result.outcome === 'failed' ? 'Interpretation failed.' : result.outcome === 'cancelled' ? 'Interpretation stopped.' : result.stop === 'token_limit' ? 'Output budget reached.' : 'Interpretation complete.';
    if (result.format !== 'complete') status.textContent += ' Showing unfinished or unstructured NLA output.';
    article.append(title, excerpt, text, status);
    const details = document.createElement('details'), detailSummary = document.createElement('summary'), metadata = document.createElement('p');
    detailSummary.textContent = 'Generation details'; metadata.className = 'hint';
    metadata.textContent = `Block ${result.capture.block} · seed ${result.settings.seed} · temperature ${result.settings.temperature} · ${result.format} explanation tags`;
    details.append(detailSummary, metadata);
    if (result.stream) { const raw = document.createElement('pre'); raw.textContent = new TextDecoder().decode(Uint8Array.from(result.stream.generatedBytes)); details.append(raw); }
    article.append(details);
    $('interpretations').prepend(article);
  }
  renderOutputState();
  renderLiveInterpretation();
  renderSelection();
  if (state.status?.publication_failed) showError(new Error('Live publication failed. The displayed transcript may be incomplete; start a new experiment after the operation stops.'));
}
function settings(prefix) {
  return generationSettings({
    mode: $(prefix + '-mode').value,
    seed: $(prefix + '-seed').value,
    max_new_tokens: Number($(prefix + '-cap').value),
    temperature: Number($(prefix + '-temperature').value),
    top_p: prefix === 'av' ? 1 : Number($('target-top-p').value)
  });
}
async function command(path, body, requestEpoch = state.epoch) {
  const retry = () => command(path, body, requestEpoch);
  try { clearError(); submitting = true; render(); const receipt = await client.command(path, body, requestEpoch);
    if (body.prompt) {
      $('composer').open = false; $('prompt-preview').textContent = body.prompt.user;
      try { sessionStorage.setItem('dolittle-prompt', JSON.stringify({ epoch: state.epoch, run: receipt.run, prompt: body.prompt, settings: body.settings })); } catch { /* A private browser may disable storage. */ }
    }
    if (body.jobs?.length) pendingInterpretation = { request: body.request, text: selectedText(state, body.jobs[0].capture.position) };
    if (body.jobs?.length && $('av-seed').value === body.jobs[0].settings.seed) {
      const next = BigInt(body.jobs[0].settings.seed) + BigInt(body.jobs.length);
      if (next <= 18446744073709551615n) $('av-seed').value = next.toString();
    }
    if (receipt.run !== state.run) { pendingInterpretation = null; state = newState(state.epoch, receipt.run); resetView(); } await poll(); }
  catch (error) { showError(error, error.status ? null : retry); }
  finally { submitting = false; render(); }
}
let polling = false;
async function poll() {
  if (polling) return;
  polling = true;
  try {
    const status = await client.status(); connected = true;
    if (needsAccess) { clearError(); needsAccess = false; }
    if (status.epoch !== state.epoch || status.run !== state.run) { pendingInterpretation = null; state = newState(status.epoch, status.run); resetView(); }
    if (!restored) {
      restored = true;
      if (status.run) {
        $('composer').open = false;
        $('prompt-preview').textContent = 'Retained experiment · open to start another prompt';
        try {
          const saved = JSON.parse(sessionStorage.getItem('dolittle-prompt'));
          if (saved?.epoch === status.epoch && saved.run === status.run) {
            $('prompt').value = saved.prompt.user; $('system').value = saved.prompt.system ?? '';
            $('thinking').checked = saved.prompt.thinking === 'enabled';
            $('target-cap').value = saved.settings.max_new_tokens;
            $('target-mode').value = saved.settings.mode;
            $('target-seed').value = saved.settings.seed;
            $('target-temperature').value = saved.settings.temperature;
            $('target-top-p').value = saved.settings.top_p;
            $('target-temperature').disabled = saved.settings.mode === 'greedy';
            $('target-top-p').disabled = saved.settings.mode === 'greedy';
            $('prompt-preview').textContent = saved.prompt.user;
          }
        } catch { /* Stored prompt context is optional. */ }
      }
    }
    capabilities = status.limits; state.status = status; $('pair').textContent = status.pair === 'qwen3-8b-nla' ? 'Qwen3-8B · Qwen3-8B-nla' : status.pair;
    if (state.run) {
      const page = await client.events(state.run, state.cursor);
      applyPage(state, page); pollDelay = page.more || state.status.active_request || state.status.queued_jobs ? 100 : 1000;
    } else pollDelay = 1000;
    render();
  } catch (error) { connected = false; pollDelay = Math.min(Math.max(pollDelay * 2, 1000), 10000); if (error.status === 401) { needsAccess = true; showError(error); } $('connection').textContent = 'Disconnected — reconnecting'; $('generate').disabled = true; $('stop').disabled = true; $('interpret').disabled = true; }
  finally { polling = false; }
}
for (const prefix of ['target', 'av']) {
  $(prefix + '-mode').addEventListener('change', () => {
    const greedy = $(prefix + '-mode').value === 'greedy';
    $(prefix + '-temperature').disabled = greedy;
    if (prefix === 'target') $('target-top-p').disabled = greedy;
  });
}
$('prompt-form').addEventListener('invalid', event => { const details = event.target.closest('details'); if (details) details.open = true; }, true);
$('prompt-form').addEventListener('submit', event => { event.preventDefault(); try { const system = $('system').value; const body = { request: requestId(), replace: state.run, prompt: { user: $('prompt').value, thinking: $('thinking').checked ? 'enabled' : 'disabled', ...(system ? { system } : {}) }, settings: settings('target') }; command('/api/v1/runs', body); } catch (error) { showError(error); } });
$('stop').addEventListener('click', () => { if (state.status?.active_request) command(`/api/v1/runs/${state.run}/stop`, { request: requestId(), active_request: state.status.active_request }); });
$('interpret').addEventListener('click', () => { try { const jobs = selectionJobs(state, Number($('position').value), Number($('end-position').value), Number($('variant-count').value), settings('av'), capabilities.jobs); command(`/api/v1/runs/${state.run}/interpretations`, { request: requestId(), jobs }); } catch (error) { showError(error); } });
$('previous').addEventListener('click', () => select(state.selection[0] - 1)); $('next').addEventListener('click', () => select(state.selection[0] + 1));
$('position').addEventListener('change', () => select(Number($('position').value))); $('end-position').addEventListener('change', () => select(Number($('position').value), Number($('end-position').value)));
$('show-controls').addEventListener('change', () => { resetView(); render(); });
$('retry').addEventListener('click', () => retryAction?.());
$('reasoning-section').addEventListener('toggle', renderSelection);
$('composer').addEventListener('toggle', () => { $('composer-action').textContent = $('composer').open ? 'Collapse' : 'Edit prompt'; });
$('variant-count').addEventListener('input', renderSelection);
$('clear-selection').addEventListener('click', () => { state.selection = null; renderSelection(); });
async function loop() { await poll(); setTimeout(loop, pollDelay); }
loop();
