import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createClient } from '../../app/client.js';

function storage() {
  const values = new Map();
  return { getItem: key => values.get(key) ?? null, setItem: (key, value) => values.set(key, value), removeItem: key => values.delete(key) };
}
function browser(localStorage = storage(), hash = '') {
  const env = {
    localStorage, sessionStorage: storage(), location: new URL('http://localhost:8787/' + hash),
    requests: [], response: 200,
    async fetch(path, options) {
      env.requests.push({ path, ...options });
      return { status: env.response, ok: env.response === 200, text: async () => JSON.stringify({ epoch: 'current' }) };
    },
    history: { replaceState(_state, _title, path) { env.location = new URL(path, env.location); } }
  };
  return env;
}

test('launcher access survives new tabs and is removed from the address bar', async () => {
  const first = browser(storage(), '#token=first-key');
  await createClient(first).status();
  assert.equal(first.location.hash, '');
  const second = browser(first.localStorage);
  await createClient(second).status();
  assert.equal(second.requests[0].headers.Authorization, 'Bearer first-key');
  const legacy = browser(); legacy.sessionStorage.setItem('dolittle-access', 'legacy-key');
  await createClient(legacy).status();
  const migrated = browser(legacy.localStorage);
  await createClient(migrated).status();
  assert.equal(migrated.requests[0].headers.Authorization, 'Bearer legacy-key');
});

test('missing and rejected access stops API retries and a fresh link recovers', async () => {
  const env = browser(), client = createClient(env);
  await assert.rejects(client.status(), { status: 401, accessReason: 'missing' });
  assert.equal(env.requests.length, 0);
  client.connect('http://localhost:8787/#token=old-key'); env.response = 401;
  await assert.rejects(client.status(), { status: 401, accessReason: 'expired' });
  await assert.rejects(client.status(), { status: 401, accessReason: 'missing' });
  assert.equal(env.requests.length, 1);
  client.connect('http://localhost:8787/#token=new-key'); env.response = 200;
  await client.status();
  assert.equal(env.requests.at(-1).headers.Authorization, 'Bearer new-key');
});

test('access links stay on the current origin and transport failures retain access', async () => {
  const env = browser(), client = createClient(env);
  for (const link of ['not a link', 'http://localhost:8787/', 'https://other.example/#token=key']) assert.throws(() => client.connect(link));
  assert.equal(env.requests.length, 0);
  client.connect('http://localhost:8787/#token=working-key');
  env.response = 503;
  await assert.rejects(client.status(), { status: 503 });
  env.response = 200;
  await client.status();
  assert.equal(env.requests.at(-1).headers.Authorization, 'Bearer working-key');
});

test('blocked storage still permits access for the page lifetime', async () => {
  const env = browser(storage(), '#token=private-key');
  for (const name of ['localStorage', 'sessionStorage']) Object.defineProperty(env, name, { get() { throw new Error('Storage disabled'); } });
  const client = createClient(env);
  await client.status(); await client.status();
  assert.equal(env.location.hash, '');
  assert.equal(env.requests.at(-1).headers.Authorization, 'Bearer private-key');
  env.response = 401;
  await assert.rejects(client.status(), { accessReason: 'expired' });
  await assert.rejects(client.status(), { accessReason: 'missing' });
});
