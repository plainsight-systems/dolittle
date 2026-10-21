// A2: exact identity, ordered event reduction and UTF-8 display mapping.
// Pure functions below run without DOM/network/GPU and are tested as units.
const decoder = new TextDecoder('utf-8', { fatal: true });
export function newState(epoch = null, run = null) {
  return { epoch, run, cursor: '0', status: null, target: stream(), attempts: new Map(), results: [], retained: null, selection: null, requests: new Map(), trials: new Map() };
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
      } else if (['baseline', 'steered'].includes(event.role)) {
        const trial = state.trials.get(event.request);
        if (!trial) throw new Error('Missing steering trial.');
        target = trial[event.role];
      } else if (event.role !== 'target') throw new Error('Unknown stream role.');
      append(target, event);
    } else if (event.type === 'target_retained') {
      flush(state.target, true); state.target.finished = true; state.retained = event;
    } else if (event.type === 'interpretation') {
      const s = state.attempts.get(`${event.request}/${event.job}`);
      if (s) { flush(s, true); s.finished = true; }
      state.results.push({ ...event, stream: s });
    } else if (event.type === 'steering_started') {
      state.trials.set(event.request, { ...event, baseline: stream(), steered: stream() });
    } else if (event.type === 'steering_complete') {
      const trial = state.trials.get(event.request);
      if (!trial) throw new Error('Missing steering trial.');
      Object.assign(trial, event);
      for (const s of [trial.baseline, trial.steered]) { flush(s, true); s.finished = true; }
    } else if (event.type === 'request_finished') {
      state.requests.set(event.request, event);
      const trial = state.trials.get(event.request);
      if (trial) {
        if (!trial.outcome) Object.assign(trial, { outcome: event.outcome, message: event.message });
        for (const s of [trial.baseline, trial.steered]) { flush(s, true); s.finished = true; }
      }
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

// A2 presentation projections preserve the reducer's byte-derived ranges.
// No model IDs or delimiter text are reclassified by these helpers.
export function sourceSection(group) {
  return group.prompt ? 'prompt' : ['reasoning', 'answer'].includes(group.kind) ? group.kind : 'unclassified';
}
export function sourceGroups(state, section, controls = false) {
  // Generated controls stay next to the segment they delimit. Their actual
  // positions remain available through the numeric navigator and token strip.
  let preceding = 'reasoning';
  return state.target.groups.filter(g => {
    if (!g.prompt && g.kind !== 'control') preceding = sourceSection(g);
    const owner = g.kind === 'control' && !g.prompt ? preceding : sourceSection(g);
    return owner === section && (controls || g.kind !== 'control');
  });
}
export function findOccurrences(groups, query) {
  if (!query) return [];
  const text = groups.map(g => g.text).join(''), matches = [];
  // Unicode regex case folding preserves original string offsets, unlike
  // lowercasing which can change the length of a code point.
  const escaped = query.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  const pattern = new RegExp(escaped, 'giu');
  const word = /^[\p{L}\p{N}_]+$/u.test(query), offsets = [];
  let offset = 0;
  for (const g of groups) { offsets.push({ start: offset, end: offset + g.text.length, g }); offset += g.text.length; }
  for (const match of text.matchAll(pattern)) {
    const begin = match.index, end = begin + match[0].length;
    const before = [...text.slice(0, begin)].at(-1) ?? '', after = [...text.slice(end)][0] ?? '';
    if (word && (/[\p{L}\p{N}_]/u.test(before) || /[\p{L}\p{N}_]/u.test(after))) continue;
    const contributors = offsets.filter(o => o.end > begin && o.start < end);
    if (contributors.length) matches.push([contributors[0].g.first, contributors.at(-1).g.last]);
  }
  return matches;
}
export function captureKey(capture) { return `${capture.block}/${capture.boundary}/${capture.position}`; }
export function samplesAt(samples, capture) {
  return samples.filter(r => captureKey(r.capture) === captureKey(capture));
}
// Submitted job metadata can supplement the journal during streaming. A terminal
// server result always wins. Never attach a stream to the current UI selection.
export function investigationSamples(state, submitted = []) {
  const samples = state.results.map(result => ({ ...result, key: `${result.request}/${result.job}`, pending: false }));
  const completed = new Set(samples.map(s => s.key));
  for (const request of submitted) {
    if (request.epoch !== state.epoch || request.run !== state.run) continue;
    request.jobs.forEach((job, index) => {
      const key = `${request.request}/${index}`;
      if (completed.has(key)) return;
      const terminal = state.requests.get(request.request);
      const stream = state.attempts.get(key);
      samples.push({ ...job, request: request.request, job: index, key, stream, pending: !terminal,
        outcome: terminal?.outcome, message: terminal?.message, format: 'missing' });
    });
  }
  return samples;
}

// The submitted user text is optional presentation provenance, never inferred
// from delimiter spellings. Only a unique, whole-group match may hide template
// text. Ambiguous/missing provenance falls back to the exact retained input.
export function promptExcerpt(state, user) {
  if (typeof user !== 'string' || !user.length) return null;
  const groups = state.target.groups.filter(g => g.prompt), text = groups.map(g => g.text).join('');
  const start = text.indexOf(user);
  if (start < 0 || text.indexOf(user, start + 1) !== -1) return null;
  let offset = 0;
  const selected = [];
  for (const g of groups) {
    const end = offset + g.text.length;
    if (end > start && offset < start + user.length) {
      if (offset < start || end > start + user.length) return null;
      selected.push(g);
    }
    offset = end;
  }
  return selected.length ? selected : null;
}

export function nextSampleSeed(samples, initial = '100') {
  const used = new Set(samples.map(s => s.settings?.seed).filter(s => /^(0|[1-9]\d*)$/.test(s)));
  let next = BigInt(initial);
  for (const seed of used) if (BigInt(seed) >= next) next = BigInt(seed) + 1n;
  if (next > 18446744073709551615n) { next = 0n; while (used.has(next.toString())) next++; }
  return next.toString();
}

export function canSteer(sample) {
  if (!sample?.variant || sample.outcome !== 'completed' || sample.format !== 'complete' || !sample.content || !sample.stream) return false;
  try {
    const bytes = Uint8Array.from(sample.stream.generatedBytes).subarray(...sample.content);
    return new TextDecoder('utf-8', {fatal:true}).decode(bytes).trim().length > 0;
  } catch { return false; }
}
// Prefix tokens are delivered for exact reconnect state, but shown once above
// the branches. Continuation text uses server segment labels and byte mapping.
export function continuationText(trial, branch, kind) {
  const s = trial[branch];
  return s.groups.filter(g => g.last >= trial.prefix_tokens && g.kind === kind).map(g => g.text).join('');
}
// One contiguous edit range preserves exact text and handles Unicode boundaries.
export function editedRange(original, edited) {
  const a = [...original], b = [...edited]; let first = 0, tail = 0;
  while (first < a.length && first < b.length && a[first] === b[first]) first++;
  while (tail < a.length - first && tail < b.length - first && a[a.length-1-tail] === b[b.length-1-tail]) tail++;
  return [b.slice(0, first).join(''), b.slice(first, b.length-tail).join(''), b.slice(b.length-tail).join('')];
}
