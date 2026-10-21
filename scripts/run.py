#!/usr/bin/env python3
"""Build Dolittle from source, verify models and launch its browser service.

Default: Linux/NVIDIA through Docker Compose; macOS/Apple Silicon builds natively.
--native-binary selects an existing source-built developer binary.
The host owns driver/toolkit installation. Never download application binaries.
"""
import argparse
import json
import os
import platform
from pathlib import Path
import secrets
import subprocess
import sys
import tempfile
import time
import urllib.request
import webbrowser

import models

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, default=ROOT / 'models')
    parser.add_argument('--port', type=int, default=8787)
    parser.add_argument('--bind', default='127.0.0.1', help='Published host address; explicit LAN opt-in')
    parser.add_argument('--host', help='Browser hostname; required with a non-loopback bind')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--verify-models', action='store_true', help='Force full model hash verification and repair before launch')
    parser.add_argument('--no-browser', action='store_true')
    parser.add_argument('--native-binary', type=Path)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535 or args.device < 0:
        parser.error('Invalid port/device')
    if args.bind not in {'127.0.0.1', 'localhost'} and not args.host:
        parser.error('--host is required for an explicit LAN bind')
    host = args.host or '127.0.0.1'
    if any(c in host for c in '/\\?#@\r\n'):
        parser.error('Invalid browser host')
    address = f'{host}:{args.port}'
    cache = args.cache.absolute()
    native_binary = args.native_binary
    if native_binary is None and sys.platform == 'darwin':
        if platform.machine() != 'arm64' or int(platform.mac_ver()[0].split('.')[0]) < 26:
            parser.error('Native Metal requires Apple Silicon and macOS 26 or later.')
        if args.device != 0:
            parser.error('Metal uses the default Apple GPU (--device 0).')
        print('Building native Metal application from source...', flush=True)
        subprocess.run(['cmake', '--preset', 'metal-release'], cwd=ROOT, check=True)
        subprocess.run(['cmake', '--build', '--preset', 'metal-release', '--target', 'dolittle', '--parallel', str(min(os.cpu_count() or 2, 8))], cwd=ROOT, check=True)
        native_binary = ROOT / 'build/metal-release/dolittle'
    if native_binary is None:
        if sys.platform != 'linux':
            parser.error('Supported launch hosts: Linux/NVIDIA or macOS/Apple Silicon.')
        subprocess.run(['docker', 'compose', 'version'], check=True)
        info = json.loads(subprocess.check_output(['docker', 'info', '--format', '{{json .}}']))
        if 'nvidia' not in info.get('Runtimes', {}):
            parser.error('NVIDIA Container Toolkit must be configured on the host before launch: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html')
        subprocess.run(['docker', 'build', '--target', 'runtime', '-t', 'dolittle-local:runtime', '.'], cwd=ROOT, check=True)
        subprocess.run(['docker', 'run', '--rm', '--gpus', f'device={args.device}', 'dolittle-local:runtime', 'nvidia-smi', '-L'], check=True)
    # Setup verifies once; unchanged installations use their persistent receipt.
    models.prepare(ROOT / 'config/models.json', cache, args.offline, force=args.verify_models)
    local = ROOT / '.local'
    local.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='launch-', dir=local) as directory:
        token_file = Path(directory) / 'access-token'
        token = secrets.token_urlsafe(32)
        token_file.write_text(token)
        token_file.chmod(0o600)
        env = os.environ.copy()
        if native_binary:
            command = [str(native_binary.absolute()), '--manifest', str(ROOT / 'config/models.json'), '--cache', str(cache), '--assets', str(ROOT / 'app'), '--token-file', str(token_file), '--bind', args.bind, '--port', str(args.port), '--host', address, '--device', str(args.device), '--model-integrity', 'prepared']
        else:
            env.update(DOLITTLE_CACHE=str(cache), DOLITTLE_TOKEN_FILE=str(token_file), DOLITTLE_BIND=args.bind, DOLITTLE_PORT=str(args.port), DOLITTLE_HOST=address, DOLITTLE_DEVICE=str(args.device), DOLITTLE_MODEL_INTEGRITY='prepared')
            command = ['docker', 'compose', 'up', '--no-build', '--abort-on-container-exit']
        process = subprocess.Popen(command, cwd=ROOT, env=env)
        try:
            url = f'http://{address}'
            # Setup completed before spawning; startup maps and parses metadata.
            for _ in range(1200):
                if process.poll() is not None:
                    raise RuntimeError('Service exited before readiness')
                try:
                    request = urllib.request.Request(url + '/api/v1/status', headers={'Authorization': 'Bearer ' + token})
                    with urllib.request.urlopen(request, timeout=1) as response:
                        json.load(response)
                    break
                except OSError:
                    time.sleep(0.25)
            else:
                raise RuntimeError('Service did not become ready')
            access = url + '/#token=' + token
            print('Open Dolittle — use this full access link the first time in each browser:\n' + access, flush=True)
            print('After connecting, this browser remembers access at ' + url + '/\n'
                  'Keep this terminal running. If you restart the launcher, open its new access link.', flush=True)
            if not args.no_browser:
                webbrowser.open(access)
            return process.wait()
        except KeyboardInterrupt:
            return 0
        finally:
            if native_binary:
                process.terminate()
                process.wait()
            else:
                subprocess.run(['docker', 'compose', 'down', '--timeout', '120'], cwd=ROOT, env=env, check=False)
                process.wait()


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        raise SystemExit(f'dolittle: {error}') from error
