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

test('search preserves positions across subwords and split Unicode without partial word matches', async () => {
  const { findOccurrences } = await import('../../app/controller.js');
  const groups = [
    { first: 0, last: 0, text: 'with ' }, { first: 1, last: 1, text: 'w' },
    { first: 2, last: 2, text: 'it ' }, { first: 3, last: 4, text: '€' },
    { first: 5, last: 5, text: ' WIT. [x]' }
  ];
  assert.deepEqual(findOccurrences(groups, 'wit'), [[1, 2], [5, 5]]);
  assert.deepEqual(findOccurrences(groups, '€'), [[3, 4]]);
  assert.deepEqual(findOccurrences(groups, '[x]'), [[5, 5]]);
  assert.deepEqual(findOccurrences(groups, ''), []);
});
test('samples keep full capture identity and pending jobs retain their submitted point', async () => {
  const { samplesAt, investigationSamples } = await import('../../app/controller.js');
  const state = newState('e', '1'), capture = { position: 8, block: 24, boundary: 'block_output' };
  const complete = { request: 'r', job: 0, capture };
  state.results = [complete, { request: 'q', job: 0, capture: { ...capture, block: 7 } }];
  assert.deepEqual(samplesAt(state.results, capture), [complete]);
  state.selection = [99, 99];
  const requests = [{ epoch: 'e', run: '1', request: 'r', jobs: [{ capture }, { capture: { ...capture, position: 9 } }] },
    { epoch: 'old', run: '1', request: 'stale', jobs: [{ capture }] }];
  let samples = investigationSamples(state, requests);
  assert.equal(samples.length, 3);
  assert.equal(samples[2].capture.position, 9);
  assert.equal(samples[2].pending, true);
  state.requests.set('r', { outcome: 'cancelled' });
  samples = investigationSamples(state, requests);
  assert.equal(samples[2].pending, false);
  assert.equal(samples[2].outcome, 'cancelled');
});
test('optional control tokens retain their sequence location and all text is preserved', async () => {
  const { sourceGroups } = await import('../../app/controller.js');
  const state = newState();
  state.target.groups = [
    { prompt: true, kind: 'unclassified', text: 'input' },
    { prompt: false, kind: 'control', text: '<think>' },
    { prompt: false, kind: 'reasoning', text: 'a\n b' },
    { prompt: false, kind: 'control', text: '</think>' },
    { prompt: false, kind: 'answer', text: 'reply' },
    { prompt: false, kind: 'control', text: '<end>' }
  ];
  assert.equal(sourceGroups(state, 'reasoning').map(g => g.text).join(''), 'a\n b');
  assert.equal(sourceGroups(state, 'reasoning', true).map(g => g.text).join(''), '<think>a\n b</think>');
  assert.equal(sourceGroups(state, 'answer', true).map(g => g.text).join(''), 'reply<end>');
});
test('prompt presentation requires unambiguous provenance on exact token boundaries', async () => {
  const { promptExcerpt } = await import('../../app/controller.js');
  const state = newState();
  state.target.groups = [{ prompt: true, text: 'prefix' }, { prompt: true, text: 'hello' }, { prompt: true, text: 'suffix' }];
  assert.deepEqual(promptExcerpt(state, 'hello'), [state.target.groups[1]]);
  assert.equal(promptExcerpt(state, 'ell'), null);
  assert.equal(promptExcerpt(state, undefined), null);
  state.target.groups.push({ prompt: true, text: 'hello' });
  assert.equal(promptExcerpt(state, 'hello'), null);
});
test('new samples advance past retained seeds without precision loss or overflow', async () => {
  const { nextSampleSeed } = await import('../../app/controller.js');
  assert.equal(nextSampleSeed([]), '100');
  assert.equal(nextSampleSeed([{ settings: { seed: '104' } }]), '105');
  assert.equal(nextSampleSeed([{ settings: { seed: '9007199254740993' } }]), '9007199254740994');
  assert.equal(nextSampleSeed([{ settings: { seed: '18446744073709551615' } }, { settings: { seed: '0' } }]), '1');
});
test('steering journal separates branches and replays without altering the original', async () => {
  const { continuationText } = await import('../../app/controller.js');
  const s = newState('e', '1');
  const start = {sequence:'1',type:'steering_started',request:'trial',variant:'7',capture:{position:0,block:24,boundary:'block_output'},prefix_tokens:1,original_prompt_tokens:1};
  const branch = (seq, role, first, char) => ({...token(seq,first,[char.charCodeAt(0)]),request:'trial',role,prompt_tokens:1});
  const events = [start,branch('2','baseline',0,'P'),branch('3','baseline',1,'B'),branch('4','steered',0,'P'),branch('5','steered',1,'S'),{sequence:'6',type:'steering_complete',request:'trial',trial:'1',outcome:'completed'}];
  const p = page(events,'6'); applyPage(s,p); applyPage(s,p);
  assert.equal(s.target.tokens.length,0);
  assert.equal(s.trials.size,1);
  assert.equal(continuationText(s.trials.get('trial'),'baseline','reasoning'),'B');
  assert.equal(continuationText(s.trials.get('trial'),'steered','reasoning'),'S');
});
test('steering source requires completed delimited nonempty text; edit ranges preserve Unicode', async () => {
  const {canSteer,editedRange} = await import('../../app/controller.js');
  const sample={variant:'1',outcome:'completed',format:'complete',content:[0,1],stream:{generatedBytes:[65]}};
  assert.equal(canSteer(sample),true);
  assert.equal(canSteer({...sample,outcome:'cancelled'}),false);
  assert.equal(canSteer({...sample,format:'missing'}),false);
  assert.equal(canSteer({...sample,stream:{generatedBytes:[0xff]}}),false);
  assert.deepEqual(editedRange('🐇 carrot','🐁 carrot'),['','🐁',' carrot']);
});
