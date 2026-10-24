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

test('invalid launcher links are rejected and transport failures retain access', async () => {
  const env = browser(), client = createClient(env);
  for (const link of ['not a link', 'http://localhost:8787/', 'file:///tmp/key#token=key', 'javascript:alert(1)#token=key', 'https://user:password@other.example/#token=key']) assert.throws(() => client.connect(link));
  assert.equal(env.requests.length, 0);
  client.connect('http://localhost:8787/#token=working-key');
  env.response = 503;
  await assert.rejects(client.status(), { status: 503 });
  env.response = 200;
  await client.status();
  assert.equal(env.requests.at(-1).headers.Authorization, 'Bearer working-key');
});

test('remote launcher links authorize only the currently opened forwarded page', async () => {
  for (const address of ['http://127.0.0.1:8877/', 'http://localhost:8877/', 'http://[::1]:8877/']) {
    const env = browser(); env.location = new URL(address);
    const client = createClient(env);
    client.connect('http://spark.example:8787/#token=forwarded-key');
    assert.equal(env.location.href, address);
    assert.equal(env.requests.length, 0);
    await client.status();
    assert.equal(new URL(env.requests[0].path, env.location).origin, env.location.origin);
    assert.equal(env.requests[0].headers.Authorization, 'Bearer forwarded-key');
    const reopened = browser(env.localStorage); reopened.location = new URL(address);
    await createClient(reopened).status();
    assert.equal(reopened.requests[0].headers.Authorization, 'Bearer forwarded-key');
  }
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
