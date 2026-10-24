#!/usr/bin/env python3
"""One real source-launch → capture → interpretation → reconnect interaction.

Requires the pinned revision cache and a GPU. Select a source-built native
executable or the default Docker launcher with host NVIDIA Container Toolkit.
No model download, mock engine, fixture rewriting or semantic prose assertions.
This replaces the standalone archive scenario; setup verifies files and launch
reuses the prepared installation. Exercise that lifecycle before the GPU flow.
"""
import argparse
import json
import hashlib
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(ROOT / 'scripts'))
import models


def preparation_lifecycle():
    # The source-launch interaction starts with an admitted model installation.
    # Tiny real files exercise invalidation without damaging the pinned cache.
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        payload = b'verified model input'
        data = {'schema': 1, 'pair': 'qwen3-8b-nla', 'models': [
            {'role': role, 'repo': 'test/' + role, 'revision': 'a' * 40,
             'index': 'model.safetensors',
             'files': [{'path': 'model.safetensors', 'bytes': len(payload),
                        'sha256': hashlib.sha256(payload).hexdigest()}]}
            for role in ('target', 'verbalizer')]}
        source, cache = root / 'manifest.json', root / 'cache'
        source.write_text(json.dumps(data))
        paths = []
        for model in data['models']:
            path = models.destination(cache, model, model['files'][0])
            path.parent.mkdir(parents=True)
            path.write_bytes(payload)
            paths.append(path)
        models.prepare(source, cache, offline=True)
        receipt_path = cache / models.RECEIPT
        receipt = receipt_path.read_bytes()
        models.prepare(source, cache, offline=True)
        assert receipt_path.read_bytes() == receipt
        paths[0].write_bytes(b'X' * len(payload))
        try:
            models.prepare(source, cache, offline=True)
        except ValueError:
            pass
        else:
            raise AssertionError('Changed corrupt installation admitted')
        assert not receipt_path.exists(), 'Failed setup retained a receipt'
        paths[0].write_bytes(payload)
        models.prepare(source, cache, offline=True)
        models.prepare(source, cache, offline=True, force=True)
        paths[1].unlink()
        try:
            models.prepare(source, cache, offline=True)
        except ValueError:
            pass
        else:
            raise AssertionError('Incomplete installation admitted')
        assert not receipt_path.exists()


def run(binary, cache, port, report, hold=False):
    preparation_lifecycle()
    report.mkdir(parents=True, exist_ok=True)
    log_path = report / 'service.log'
    command = ['python3', 'scripts/run.py', '--offline', '--no-browser',
               '--cache', str(cache.resolve()), '--port', str(port)]
    if binary is not None:
        command.extend(['--native-binary', str(binary.resolve())])
    with log_path.open('w+') as log:
        process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=log,
                                   start_new_session=True)
        try:
            deadline = time.monotonic() + 600
            access = None
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError('Launcher failed; inspect service.log')
                match = re.search(r'http://127\.0\.0\.1:\d+/#token=([^\s]+)', log_path.read_text())
                if match:
                    access = match[0]
                    break
                time.sleep(0.2)
            assert access, 'Launcher readiness timeout'
            parsed = urllib.parse.urlsplit(access)
            base = f'{parsed.scheme}://{parsed.netloc}'
            token = urllib.parse.parse_qs(parsed.fragment)['token'][0]
            epoch = None

            def request(path, body=None, authenticated=True, expected=200, extra_headers=None):
                headers = {'Authorization': 'Bearer ' + token} if authenticated else {}
                if epoch:
                    headers['X-Dolittle-Epoch'] = epoch
                data = json.dumps(body).encode() if body else None
                if data:
                    headers['Content-Type'] = 'application/json'
                headers.update(extra_headers or {})
                req = urllib.request.Request(base + path, data=data, headers=headers)
                try:
                    response = urllib.request.urlopen(req, timeout=15)
                except urllib.error.HTTPError as error:
                    response = error
                with response:
                    assert response.status == expected, (path, response.status, response.read().decode())
                    content = response.read()
                    return json.loads(content) if response.headers.get_content_type() == 'application/json' else content

            request('/api/v1/status', authenticated=False, expected=401)
            for authority in ('localhost:8877', '127.0.0.1:8877', '[::1]:8877'):
                forwarded = {'Host': authority, 'Origin': 'http://' + authority}
                request('/', extra_headers=forwarded)
                request('/api/v1/status', extra_headers=forwarded)
                request('/api/v1/status', authenticated=False, expected=401, extra_headers=forwarded)
            request('/api/v1/status', expected=403, extra_headers={'Host': 'evil.example'})
            request('/api/v1/status', expected=403, extra_headers={'Origin': 'http://evil.example'})
            request('/api/v1/status', expected=403, extra_headers={'Host': 'localhost:8877', 'Origin': 'http://localhost:8878'})
            assert b'What was present before the next word?' in request('/')
            status = request('/api/v1/status')
            epoch = status['epoch']
            settings = dict(mode='greedy', temperature=1, top_k=0, top_p=1, seed='42', max_new_tokens=8)
            command = dict(request='application-target', prompt=dict(user='A rhyming couplet:\nHe saw a carrot and had to grab it,', thinking='enabled'), settings=settings)
            receipt = request('/api/v1/runs', command, expected=202)
            run_id = receipt['run']
            replay = request('/api/v1/runs', command)
            assert replay['run'] == run_id and replay['code'] == 'replayed'
            events, cursor = [], '0'

            def drain_until(request_id):
                nonlocal cursor
                deadline = time.monotonic() + 600
                while time.monotonic() < deadline:
                    page = request(f'/api/v1/runs/{run_id}/events?after={cursor}')
                    for event in page['events']:
                        assert int(event['sequence']) == int(cursor) + 1
                        cursor = event['sequence']
                        events.append(event)
                    assert cursor == page['next']
                    finished = [e for e in events if e['type'] == 'request_finished' and e['request'] == request_id]
                    if finished:
                        assert finished[-1]['outcome'] == 'completed', finished[-1]
                        return
                    time.sleep(0.1)
                raise AssertionError('Request completion timeout')

            drain_until(command['request'])
            retained = next(e for e in events if e['type'] == 'target_retained')
            target_deltas = [e for e in events if e['type'] == 'tokens' and e['role'] == 'target']
            assert target_deltas and retained['capture_ranges']
            prompt_count = target_deltas[0]['prompt_tokens']
            assert any(begin <= prompt_count < end for begin, end in retained['capture_ranges'])
            import base64
            selected = [e['first']+i for e in target_deltas for i,t in enumerate(e['tokens'])
                        if e['first']+i < prompt_count and base64.b64decode(e['bytes'])[t['begin']:t['end']].strip()==b'it']
            assert selected, 'Paper prompt token is missing'
            capture = {**retained['site'], 'position': selected[-1]}
            interpretation = dict(request='application-interpretation', jobs=[dict(capture=capture, settings={**settings, 'seed':'100', 'max_new_tokens':256})])
            path = f'/api/v1/runs/{run_id}/interpretations'
            request(path, interpretation, expected=202)
            drain_until(interpretation['request'])
            done = [e for e in events if e['type'] == 'interpretation']
            assert len(done) == 1 and done[0]['outcome'] == 'completed' and done[0]['variant']
            assert any(e['type'] == 'tokens' and e['role'] == 'verbalizer' and e['tokens'] for e in events)
            assert done[0]['format'] == 'complete' and done[0]['content'], done[0]
            import base64
            raw = b''.join(base64.b64decode(e['bytes'])[t['begin']:t['end']]
                           for e in events if e['type']=='tokens' and e['role']=='verbalizer'
                           for i,t in enumerate(e['tokens']) if e['first']+i >= e['prompt_tokens'])
            (report / 'events-before-steering.json').write_text(json.dumps(events))
            (report / 'sample.bin').write_bytes(raw)
            original = raw[slice(*done[0]['content'])].decode()
            steering = dict(request='application-steering',variant=done[0]['variant'],
                            explanation=original+' The next answer discusses a rabbit.',strength=0.5,
                            settings={**settings,'max_new_tokens':8})
            steering_path = f'/api/v1/runs/{run_id}/steering'
            request(steering_path,steering,expected=202)
            drain_until(steering['request'])
            trial = next(e for e in events if e['type']=='steering_complete')
            assert trial['outcome']=='completed',trial
            assert request(steering_path,steering)['code']=='replayed'
            for branch in ('baseline','steered'):
                assert any(e['type']=='tokens' and e['role']==branch and e['tokens'] for e in events)
            control = {**steering, 'request':'application-zero-control', 'explanation':original, 'strength':0,
                       'settings':{**settings,'mode':'categorical','temperature':0.6,'top_p':0.95,'seed':'77'}}
            request(steering_path,control,expected=202)
            drain_until(control['request'])
            control_done = next(e for e in events if e['type']=='steering_complete' and e['request']==control['request'])
            assert control_done['outcome']=='completed',control_done
            def continuation_ids(branch):
                return [t['id'] for e in events if e['type']=='tokens' and e['request']==control['request'] and e['role']==branch
                        for i,t in enumerate(e['tokens']) if e['first']+i >= e['prompt_tokens']]
            assert continuation_ids('baseline') == continuation_ids('steered') and continuation_ids('baseline'), 'Zero-strength sampled branches diverged'
            request(steering_path,{**control,'strength':0.5},expected=409)
            # A lost mutation response and a new reader must not execute a variant twice.
            assert request(path, interpretation)['code'] == 'replayed'
            recovered, after = [], '0'
            while True:
                page = request(f'/api/v1/runs/{run_id}/events?after={after}')
                recovered.extend(page['events'])
                after = page['next']
                if not page['more']:
                    break
            assert recovered == events and after == cursor
            summary = dict(execution='native' if binary is not None or sys.platform == 'darwin' else 'container', backend=status['backend'],
                           run=run_id, events=len(events), target_tokens=sum(len(e['tokens']) for e in target_deltas),
                           interpretations=len(done), steering_trials=2, exact_reconnect=True, idempotent_retry=True)
            (report / 'result.json').write_text(json.dumps(summary, indent=2) + '\n')
            print(json.dumps(summary), flush=True)
            if hold:
                print('Application flow passed; holding the same experiment for browser inspection. Stop with Ctrl-C.', flush=True)
                try:
                    while True:
                        time.sleep(1)
                except KeyboardInterrupt:
                    pass
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGINT)
            try:
                process.wait(timeout=120)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
                raise AssertionError('Service failed to drain shutdown')
            assert process.returncode == 0, f'Launcher exit: {process.returncode}'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    execution = parser.add_mutually_exclusive_group(required=True)
    execution.add_argument('--binary', type=Path, help='Source-built CUDA or Metal executable')
    execution.add_argument('--native-source', action='store_true', help='Build and run the native Metal application through the source launcher')
    execution.add_argument('--container', action='store_true', help='Build and run through Docker Compose')
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--port', type=int, default=8787)
    parser.add_argument('--report', type=Path, default=ROOT / '.local/application-check')
    parser.add_argument('--hold', action='store_true', help='Keep this experiment open for browser inspection until Ctrl-C')
    args = parser.parse_args()
    if args.native_source and sys.platform != 'darwin':
        parser.error('--native-source requires macOS')
    if args.container and sys.platform != 'linux':
        parser.error('--container requires Linux')
    run(args.binary, args.cache, args.port, args.report, args.hold)
