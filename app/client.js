// A1: same-origin transport. Mutations retain their request ID across retries;
// cursor/epoch reconciliation belongs to the controller. No inference state here.
export function requestId() {
  return Array.from(crypto.getRandomValues(new Uint8Array(16)), b => b.toString(16).padStart(2, '0')).join('');
}
export function createClient() {
  function accessToken() {
    const fragment = new URLSearchParams(location.hash.slice(1));
    const supplied = fragment.get('token');
    if (supplied) { sessionStorage.setItem('dolittle-access', supplied); history.replaceState(null, '', location.pathname); }
    return sessionStorage.getItem('dolittle-access');
  }
  let epoch = null;
  async function fetchJSON(path, body, requestEpoch = epoch) {
    const headers = { Authorization: `Bearer ${accessToken() ?? ''}` };
    if (requestEpoch) headers['X-Dolittle-Epoch'] = requestEpoch;
    if (body) headers['Content-Type'] = 'application/json';
    const response = await fetch(path, { method: body ? 'POST' : 'GET', headers, body: body ? JSON.stringify(body) : undefined, cache: 'no-store', signal: AbortSignal.timeout(10000) });
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
