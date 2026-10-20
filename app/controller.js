// A2: exact identity, ordered event reduction and UTF-8 display mapping.
// Pure functions below run without DOM/network/GPU and are tested as units.
const decoder = new TextDecoder('utf-8', { fatal: true });
export function newState(epoch = null, run = null) {
  return { epoch, run, cursor: '0', status: null, target: stream(), attempts: new Map(), results: [], retained: null, selection: null, requests: new Map() };
}
function stream() { return { tokens: [], groups: [], pending: [], generatedBytes: [], promptTokens: 0, consumed: 0, finished: false }; }
function group(s, text, contributors, kind, prompt) {
  const first = contributors[0].position, last = contributors.at(-1).position;
  const previous = s.groups.at(-1);
  if (previous && previous.first === first && previous.last === last && previous.kind === kind && previous.prompt === prompt) previous.text += text;
  else s.groups.push({ text, first, last, kind, prompt });
}
function flush(s, final = false) {
  while (s.pending.length) {
    const first = s.pending[0], b = first.byte;
    const count = b < 0x80 ? 1 : b >= 0xc2 && b <= 0xdf ? 2 : b >= 0xe0 && b <= 0xef ? 3 : b >= 0xf0 && b <= 0xf4 ? 4 : 0;
    const continuation = s.pending.slice(1, count).every(p => p.byte >= 0x80 && p.byte <= 0xbf);
    if (count && continuation && s.pending.length < count && !final) return;
    let length = count, text;
    try {
      if (!count || !continuation || s.pending.length < count) throw new Error('incomplete');
      text = decoder.decode(Uint8Array.from(s.pending.slice(0, count), p => p.byte));
    } catch { length = 1; text = '\ufffd'; }
    const contributors = s.pending.splice(0, length);
    group(s, text, contributors, first.kind, first.prompt);
  }
}
function append(s, event) {
  if (!Number.isSafeInteger(event.first) || event.first !== s.tokens.length || !Number.isSafeInteger(event.consumed_tokens) || event.consumed_tokens < s.consumed || event.consumed_tokens > event.first + event.tokens.length) throw new Error('Invalid token progress.');
  const bytes = Uint8Array.from(atob(event.bytes), c => c.charCodeAt(0));
  s.promptTokens = event.prompt_tokens; s.consumed = event.consumed_tokens;
  for (const t of event.tokens) {
    if (!Number.isSafeInteger(t.id) || !Number.isSafeInteger(t.begin) || !Number.isSafeInteger(t.end) || t.begin < 0 || t.end < t.begin || t.end > bytes.length) throw new Error('Invalid token byte mapping.');
    const position = s.tokens.length, prompt = position < s.promptTokens;
    const token = { ...t, position, bytes: bytes.slice(t.begin, t.end) }; s.tokens.push(token);
    if (!prompt) for (const byte of token.bytes) s.generatedBytes.push(byte);
    if (!token.bytes.length) group(s, t.mapped ? '[empty]' : '[unmapped]', [{ position }], 'control', prompt);
    for (const byte of token.bytes) { s.pending.push({ byte, position, kind: t.kind, prompt }); flush(s); }
  }
}
export function applyPage(state, page) {
  if (page.epoch !== state.epoch || page.run !== state.run) throw new Error('Stale experiment page.');
  for (const event of page.events) {
    if (!/^\d+$/.test(event.sequence)) throw new Error('Invalid cursor.');
    if (BigInt(event.sequence) <= BigInt(state.cursor)) continue;
    if (BigInt(event.sequence) !== BigInt(state.cursor) + 1n) throw new Error('Missing experiment event.');
    if (event.type === 'tokens') {
      let target = state.target;
      if (event.role === 'target' && event.site) state.site = event.site;
      if (event.role === 'verbalizer') {
        const key = `${event.request}/${event.job}`;
        if (!state.attempts.has(key)) state.attempts.set(key, stream());
        target = state.attempts.get(key);
      } else if (event.role !== 'target') throw new Error('Unknown stream role.');
      append(target, event);
    } else if (event.type === 'target_retained') {
      flush(state.target, true); state.target.finished = true; state.retained = event;
    } else if (event.type === 'interpretation') {
      const s = state.attempts.get(`${event.request}/${event.job}`);
      if (s) { flush(s, true); s.finished = true; }
      state.results.push({ ...event, stream: s });
    } else if (event.type === 'request_finished') {
      state.requests.set(event.request, event);
      // Even a failed request can leave an incomplete final UTF-8 sequence.
      for (const s of [state.target, ...state.attempts.values()]) if (!s.finished) { flush(s, true); s.finished = true; }
    } else if (event.type !== 'phase') throw new Error('Unknown event type.');
    state.cursor = event.sequence;
  }
  if (state.cursor !== page.next) throw new Error('Page cursor does not match its events.');
  state.status = page.status; return state;
}
export function captureAvailable(state, position) {
  return !!state.retained?.capture_ranges.some(([begin, end]) => position >= begin && position < end);
}
export function interpretationText(result) {
  if (!result.stream) return result.message || 'No output was produced.';
  const bytes = Uint8Array.from(result.stream.generatedBytes);
  if (result.format === 'complete' && result.content && result.content[0] >= 0 && result.content[1] <= bytes.length) return new TextDecoder().decode(bytes.subarray(...result.content));
  return new TextDecoder().decode(bytes);
}
export function selectionJobs(state, first, last, variants, settings, maxJobs) {
  if (![first, last, variants].every(Number.isSafeInteger) || first < 0 || last < first || last >= state.target.tokens.length || variants < 1 || (last - first + 1) * variants > maxJobs) throw new Error(`Select at most ${maxJobs} position/variant combinations.`);
  const site = state.retained?.site ?? state.site;
  if (!site) throw new Error('Capture site is not available yet.');
  const jobs = [];
  for (let position = first; position <= last; position++) {
    if (state.retained && !captureAvailable(state, position)) throw new Error(`Capture ${position} is unavailable.`);
    for (let variant = 0; variant < variants; variant++) {
      const seed = BigInt(settings.seed) + BigInt(jobs.length);
      if (seed > 18446744073709551615n) throw new Error('Seed overflow.');
      jobs.push({ capture: { position, block: site.block, boundary: site.boundary }, settings: { ...settings, seed: seed.toString() } });
    }
  }
  return jobs;
}

// Greedy has no temperature/top-p transform. Canonicalize inactive controls to
// the engine's explicit greedy contract rather than submitting stale UI values.
export function generationSettings(input) {
  const { mode, seed, max_new_tokens } = input;
  if (!['greedy', 'categorical'].includes(mode) || !/^(0|[1-9]\d*)$/.test(seed) || BigInt(seed) > 18446744073709551615n || !Number.isSafeInteger(max_new_tokens) || max_new_tokens < 1) throw new Error('Check generation mode, seed and token limit.');
  const temperature = mode === 'greedy' ? 1 : input.temperature;
  const top_p = mode === 'greedy' ? 1 : input.top_p;
  if (!Number.isFinite(temperature) || temperature <= 0 || !Number.isFinite(top_p) || top_p <= 0 || top_p > 1) throw new Error('Check temperature and top p.');
  return { mode, temperature, top_k: 0, top_p, seed, max_new_tokens };
}

// Presentation facts come from the server's segment classifications; do not
// infer reasoning from prose or reinterpret model-specific control IDs here.
export function outputSummary(state) {
  const generated = state.target.tokens.slice(state.target.promptTokens);
  return {
    generated: generated.length,
    thinking: generated.filter(t => t.kind === 'reasoning').length,
    answer: generated.filter(t => t.kind === 'answer').length,
    unclassified: generated.filter(t => t.kind === 'unclassified').length,
    stop: state.retained?.stop ?? null
  };
}
export function selectedText(state, first, last = first) {
  return state.target.groups.filter(g => g.first <= last && g.last >= first).map(g => g.text).join('');
}
