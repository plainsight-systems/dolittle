// A1: same-origin transport. Mutations retain their request ID across retries;
// cursor/epoch reconciliation belongs to the controller. Access is remembered
// per origin across tabs; blocked storage falls back to this page's lifetime.
// Rejected credentials are discarded, not retried. No inference state here.
export function requestId() {
  return Array.from(crypto.getRandomValues(new Uint8Array(16)), b => b.toString(16).padStart(2, '0')).join('');
}
export function createClient(browser = globalThis) {
  const key = 'dolittle-access';
  let memoryToken = null;
  function stored(storage) { try { return browser[storage].getItem(key); } catch { return null; } }
  function remember(token) {
    memoryToken = token;
    try { browser.localStorage.setItem(key, token); } catch { /* Use memory when storage is disabled. */ }
    try { browser.sessionStorage.removeItem(key); } catch { /* Legacy storage is optional. */ }
  }
  function accessToken() {
    const fragment = new URLSearchParams(browser.location.hash.slice(1));
    const supplied = fragment.get('token');
    if (supplied) {
      remember(supplied);
      browser.history.replaceState(null, '', browser.location.pathname + browser.location.search);
    }
    const saved = stored('localStorage');
    if (saved) return saved;
    const legacy = stored('sessionStorage');
    if (legacy) remember(legacy);
    return memoryToken;
  }
  function accessError(reason) {
    const error = new Error(reason === 'expired' ? 'This access link is no longer valid. Use the link from the current launcher.' : 'Connect using the access link printed in the launcher terminal.');
    error.status = 401; error.accessReason = reason;
    return error;
  }
  let epoch = null;
  async function fetchJSON(path, body, requestEpoch = epoch) {
    const token = accessToken();
    if (!token) throw accessError('missing');
    const headers = { Authorization: `Bearer ${token}` };
    if (requestEpoch) headers['X-Dolittle-Epoch'] = requestEpoch;
    if (body) headers['Content-Type'] = 'application/json';
    const response = await browser.fetch(path, { method: body ? 'POST' : 'GET', headers, body: body ? JSON.stringify(body) : undefined, cache: 'no-store', signal: AbortSignal.timeout(10000) });
    if (response.status === 401) {
      if (stored('localStorage') === token) { try { browser.localStorage.removeItem(key); } catch { /* Storage may be disabled. */ } }
      if (memoryToken === token) memoryToken = null;
      throw accessError('expired');
    }
    const text = await response.text();
    if (text.length > 1048576) throw new Error('Service response exceeded its limit.');
    if (!response.ok) {
      let message = text;
      try { message = `Request ${JSON.parse(text).code}.`; } catch { /* Plain-text protocol error. */ }
      const error = new Error(message); error.status = response.status; throw error;
    }
    return JSON.parse(text);
  }
  return {
    connect(link) {
      let url;
      try { url = new URL(link.trim()); } catch { throw new Error('Paste the full access link printed in the launcher terminal.'); }
      if (url.origin !== browser.location.origin) throw new Error('This link is for a different address. Open it directly in your browser, or use a link for this address.');
      const token = new URLSearchParams(url.hash.slice(1)).get('token');
      if (!token) throw new Error('This link is missing its access key. Copy the complete link, including the part after #token=.');
      remember(token);
    },
    async status() { const result = await fetchJSON('/api/v1/status'); if (typeof result.epoch !== 'string') throw new Error('Invalid service epoch.'); epoch = result.epoch; return result; },
    events(run, after) { return fetchJSON(`/api/v1/runs/${run}/events?after=${after}`); },
    async command(path, body, requestEpoch) {
      for (let attempt = 0; ; attempt++) {
        try { return await fetchJSON(path, body, requestEpoch); }
        catch (error) {
          if (error.status || attempt === 2) throw error;
          await new Promise(resolve => setTimeout(resolve, 500 * (attempt + 1)));
        }
      }
    }
  };
}
