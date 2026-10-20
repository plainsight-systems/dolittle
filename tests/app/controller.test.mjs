import { test } from 'node:test';
import assert from 'node:assert/strict';
import { newState, applyPage, captureAvailable, selectionJobs } from '../../app/controller.js';
function page(events, next = String(events.length)) { return { epoch: 'e', run: '1', events, next, status: {}, more: false }; }
function token(sequence, first, bytes) { return { sequence, request: 'r', type: 'tokens', role: 'target', job: 0, first, prompt_tokens: 0, consumed_tokens: first, bytes: Buffer.from(bytes).toString('base64'), tokens: [{ id: first + 10, begin: 0, end: bytes.length, mapped: true, kind: 'reasoning' }] }; }
test('UTF-8 character split across tokens retains all contributing positions', () => {
  const state = newState('e', '1');
  applyPage(state, page([token('1', 0, [0xe2]), token('2', 1, [0x82, 0xac])]));
  assert.equal(state.target.groups[0].text, '€');
  assert.deepEqual([state.target.groups[0].first, state.target.groups[0].last], [0, 1]);
  assert.equal(captureAvailable(state, 0), false);
});
test('replay is idempotent and a missing event is rejected', () => {
  const state = newState('e', '1'), input = page([token('1', 0, [65])]);
  applyPage(state, input); applyPage(state, input);
  assert.equal(state.target.tokens.length, 1);
  assert.throws(() => applyPage(state, page([token('3', 1, [66])], '3')), /Missing/);
});
test('window expands into separate jobs and seeds without JS precision loss', () => {
  const state = newState('e', '1');
  applyPage(state, page([token('1', 0, [65]), token('2', 1, [66])]));
  state.retained = { site: { block: 24, boundary: 'block_output' }, capture_ranges: [[0, 2]] };
  const jobs = selectionJobs(state, 0, 1, 2, { seed: '9007199254740993' }, 4);
  assert.deepEqual(jobs.map(j => j.capture.position), [0, 0, 1, 1]);
  assert.equal(jobs[3].settings.seed, '9007199254740996');
  assert.throws(() => selectionJobs(state, 0, 1, 3, { seed: '0' }, 4));
});
test('pending selection can queue against the advertised site; unavailable retained rows cannot', () => {
  const state = newState('e', '1'); applyPage(state, page([token('1', 0, [65])]));
  state.site = { block: 7, boundary: 'block_input' };
  assert.equal(selectionJobs(state, 0, 0, 1, { seed: '1' }, 1)[0].capture.block, 7);
  state.retained = { site: state.site, capture_ranges: [] };
  assert.throws(() => selectionJobs(state, 0, 0, 1, { seed: '1' }, 1), /unavailable/);
});

// The wire contract separates greedy selection from stochastic transforms.
test('sampling controls produce canonical greedy settings and preserve sampled values', async () => {
  const { generationSettings } = await import('../../app/controller.js');
  const input = { mode: 'greedy', seed: '18446744073709551615', max_new_tokens: 64, temperature: 0.6, top_p: 0.95 };
  assert.deepEqual(generationSettings(input), { ...input, temperature: 1, top_p: 1, top_k: 0 });
  assert.deepEqual(generationSettings({ ...input, mode: 'categorical' }), { ...input, mode: 'categorical', top_k: 0 });
  assert.throws(() => generationSettings({ ...input, mode: 'categorical', temperature: 0 }));
  assert.throws(() => generationSettings({ ...input, seed: '18446744073709551616' }));
});

// Output presentation must preserve the distinction between prompt, thinking
// and answer; a token-budget stop does not mean an answer was produced.
test('output summary counts generated segments without treating prompt as thinking', async () => {
  const { outputSummary } = await import('../../app/controller.js');
  const state = newState('e', '1');
  state.target.promptTokens = 2;
  state.target.tokens = [{ kind: 'unclassified' }, { kind: 'control' }, { kind: 'control' }, { kind: 'reasoning' }];
  state.retained = { stop: 'token_limit' };
  assert.deepEqual(outputSummary(state), { generated: 2, thinking: 1, answer: 0, unclassified: 0, stop: 'token_limit' });
  state.target.tokens.push({ kind: 'control' }, { kind: 'answer' });
  state.retained.stop = 'stop_token';
  assert.equal(outputSummary(state).answer, 1);
  assert.equal(outputSummary(state).thinking, 1);
});
test('selection excerpts preserve characters whose bytes span tokens', async () => {
  const { selectedText } = await import('../../app/controller.js');
  const state = newState('e', '1');
  applyPage(state, page([token('1', 0, [0xe2]), token('2', 1, [0x82, 0xac]), token('3', 2, [65])]));
  assert.equal(selectedText(state, 1), '€');
  assert.equal(selectedText(state, 0, 2), '€A');
  assert.equal(selectedText(state, 8), '');
});
