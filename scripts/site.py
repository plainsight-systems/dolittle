#!/usr/bin/env python3
"""Build Dolittle's explanation-led site with Hugo 0.154.2 and Python 3.11+.

Contract: only web-owned inputs, explicitly allowlisted docs/writeups articles,
and the named artwork enter staging. Article TOML owns its metadata. Drafts
and future dates stay out of HTML and RSS; no research/process directory mount.

Design: stage in a temporary directory, build, validate publication boundaries
and local references, then replace the last good output. Serve generated output
only, on loopback. Builds default to the configured production URL; serving uses loopback.
The deployment base may include a project subpath. No model,
engine, network fetch, or third-party Python dependency is needed. Architecture:
docs/architecture/project-site.md. Hosting/deployment is a separate operation.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from functools import partial
from html.parser import HTMLParser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path, PurePosixPath
import posixpath
from collections.abc import Mapping
import re
import shutil
import subprocess
import tempfile
import tomllib
from urllib.parse import unquote, urljoin, urlsplit
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
HUGO_VERSION = '0.154.2'
KINDS = {'explanation', 'exploration', 'milestone'}


def base_url(value: str) -> str:
    parsed = urlsplit(value)
    if (parsed.scheme not in {'http', 'https'} or not parsed.netloc or
            parsed.username or parsed.password or parsed.query or parsed.fragment or
            not value.endswith('/') or any(c.isspace() for c in value)):
        raise ValueError('Base URL must be an HTTP(S) URL ending in /, without credentials, query, or fragment.')
    return value


def source_file(relative: str) -> Path:
    path = (ROOT / relative).resolve()
    if not path.is_relative_to(ROOT.resolve()) or not path.is_file():
        raise ValueError(f'Missing or out-of-repository input: {relative}')
    return path


def articles() -> list[tuple[str, dict, str]]:
    manifest = json.loads(source_file('web/publications.json').read_text())
    if set(manifest) != {'articles'} or not isinstance(manifest['articles'], list):
        raise ValueError('Publication manifest must contain an articles list.')
    names = manifest['articles']
    if len(set(names)) != len(names):
        raise ValueError('Duplicate publication name.')
    result = []
    for slug in names:
        if not re.fullmatch(r'[a-z0-9]+(?:-[a-z0-9]+)*', slug):
            raise ValueError(f'Invalid article name: {slug}')
        source = source_file(f'docs/writeups/{slug}.md').read_text()
        if not source.startswith('+++\n') or '\n+++\n' not in source[4:]:
            raise ValueError(f'{slug}: expected TOML frontmatter.')
        header, body = source[4:].split('\n+++\n', 1)
        meta = tomllib.loads(header)
        if not all(isinstance(meta.get(k), str) and meta[k] for k in ('title', 'description', 'date', 'article_type')):
            raise ValueError(f'{slug}: title, description, date and article_type are required strings.')
        if meta['article_type'] not in KINDS:
            raise ValueError(f'{slug}: unsupported writing kind.')
        if 'draft' in meta and not isinstance(meta['draft'], bool):
            raise ValueError(f'{slug}: draft must be a boolean.')
        if meta['article_type'] == 'milestone' and (type(meta.get('milestone')) is not int or meta['milestone'] < 1):
            raise ValueError(f'{slug}: milestones require a positive milestone number.')
        # The registry controls URLs and publication time, not arbitrary mounts or aliases.
        if set(meta) - {'title', 'description', 'date', 'lastmod', 'article_type', 'milestone', 'draft'}:
            raise ValueError(f'{slug}: unsupported metadata key.')
        when = datetime.fromisoformat(meta['date'].replace('Z', '+00:00'))
        if when.tzinfo is None:
            raise ValueError(f'{slug}: date must include a time zone.')
        if 'lastmod' in meta:
            if not isinstance(meta['lastmod'], str):
                raise ValueError(f'{slug}: lastmod must be a date string.')
            updated = datetime.fromisoformat(meta['lastmod'].replace('Z', '+00:00'))
            if updated.tzinfo is None or updated < when:
                raise ValueError(f'{slug}: lastmod must include a time zone and not precede publication.')
        result.append((slug, meta, body))
    return result


def stage(project: Path) -> set[str]:
    for name in ('layouts', 'assets', 'static', 'content', 'data'):
        shutil.copytree(ROOT / 'web' / name, project / name)
    shutil.copy2(source_file('web/hugo.toml'), project / 'hugo.toml')
    artwork = project / 'assets/art/dolittle-readme.png'
    artwork.parent.mkdir(parents=True)
    shutil.copy2(source_file('docs/assets/dolittle-readme.png'), artwork)
    published = set()
    now = datetime.now(timezone.utc)
    for slug, meta, body in articles():
        when = datetime.fromisoformat(meta['date'].replace('Z', '+00:00'))
        if meta.get('draft', False) or when > now:
            continue
        bundle = project / 'content/writing' / slug
        bundle.mkdir(parents=True)
        (bundle / 'index.md').write_text(json.dumps(meta, indent=2) + '\n' + body)
        published.add(slug)
    return published


class Page(HTMLParser):
    def __init__(self, html: str):
        super().__init__(convert_charrefs=True)
        self.ids: set[str] = set()
        self.links: list[str] = []
        self.titles = self.main = self.h1 = 0
        self.meta: dict[str, str] = {}
        self.canonical = None
        self.feed(html)

    def handle_starttag(self, tag, attributes):
        attrs = dict(attributes)
        if tag == 'meta':
            key = attrs.get('property') or attrs.get('name')
            if key:
                self.meta[key] = attrs.get('content', '')
        if tag == 'link' and attrs.get('rel') == 'canonical':
            self.canonical = attrs.get('href')
        if 'id' in attrs:
            if attrs['id'] in self.ids:
                raise ValueError(f'Duplicate HTML id: {attrs["id"]}')
            self.ids.add(attrs['id'])
        self.titles += tag == 'title'
        self.main += tag == 'main'
        self.h1 += tag == 'h1'
        if tag == 'img' and 'alt' not in attrs:
            raise ValueError('Image missing alternative text.')
        for key in ('href', 'src', 'poster'):
            if attrs.get(key):
                self.links.append(attrs[key])
        for candidate in attrs.get('srcset', '').split(','):
            if candidate.strip():
                self.links.append(candidate.strip().split()[0])


def check_site_snapshot(files: Mapping[str, bytes], base: str, published: set[str]) -> None:
    """Validate an immutable in-memory publication snapshot without I/O.

    Keys are canonical relative POSIX file names; values contain file bytes.
    This is the same publication/URL/social/RSS contract used after a real build.
    Parsing creates local metadata only and never mutates the supplied snapshot.
    """
    base_url(base)
    for name, content in files.items():
        path = PurePosixPath(name)
        if (not name or path.is_absolute() or '..' in path.parts or
                path.as_posix() != name or '\\' in name or not isinstance(content, bytes)):
            raise ValueError(f'Invalid snapshot file: {name}')
    pages = {name: Page(content.decode('utf-8')) for name, content in files.items()
             if name.endswith('.html')}
    expected = {'index.html', 'writing/index.html', 'about/index.html', '404.html'} | {
        f'writing/{slug}/index.html' for slug in published}
    if set(pages) != expected:
        raise ValueError(f'Unexpected public page set: {set(pages) ^ expected}')
    origin = urlsplit(base)
    prefix = unquote(origin.path)

    def local_target(link: str, referring_url: str) -> tuple[str | None, str]:
        parsed = urlsplit(urljoin(referring_url, link))
        if parsed.scheme not in {'http', 'https'} or parsed.netloc != origin.netloc:
            return None, ''
        path = unquote(parsed.path)
        if not path.startswith(prefix):
            raise ValueError(f'Link escapes site base path: {link}')
        relative = path[len(prefix):]
        target = posixpath.normpath(relative)
        if target == '..' or target.startswith('../') or target.startswith('/'):
            raise ValueError(f'Link escapes output: {link}')
        if target == '.':
            target = 'index.html'
        elif target not in files:
            target = target.rstrip('/') + '/index.html'
        if target not in files:
            raise ValueError(f'Missing target: {link}')
        return target, unquote(parsed.fragment)

    for name, page in pages.items():
        if (page.titles, page.main, page.h1) != (1, 1, 1):
            raise ValueError(f'{name}: expected one title, main, and h1.')
        page_url = urljoin(base, name.removesuffix('index.html'))
        meta = page.meta
        required = ('og:title', 'og:description', 'og:type', 'og:site_name',
                    'og:image', 'og:image:alt', 'twitter:title', 'twitter:description',
                    'twitter:image', 'twitter:image:alt')
        if (any(not meta.get(k) for k in required) or
                meta.get('og:url') != page_url or page.canonical != page_url or
                meta.get('twitter:card') != 'summary_large_image' or
                meta.get('twitter:image') != meta.get('og:image') or
                meta.get('og:image:type') != 'image/jpeg' or
                meta.get('og:image:width') != '1200' or meta.get('og:image:height') != '400'):
            raise ValueError(f'{name}: invalid social metadata or canonical URL.')
        image_url = meta['og:image']
        if not image_url.startswith(base):
            raise ValueError(f'{name}: social image must use the site base URL.')
        social, _ = local_target(image_url, page_url)
        if social is None or PurePosixPath(social).suffix != '.jpg' or not files[social].startswith(b'\xff\xd8'):
            raise ValueError(f'{name}: social image is not a JPEG.')
        for link in page.links:
            target, fragment = local_target(link, page_url)
            if target and fragment and PurePosixPath(target).suffix == '.html':
                if fragment not in pages[target].ids:
                    raise ValueError(f'{name}: missing fragment {link}')
    for name, content in files.items():
        if name.endswith('.css'):
            css_url = urljoin(base, name)
            for link in re.findall(r'url\(["\']?([^\)"\']+)', content.decode('utf-8')):
                local_target(link, css_url)
    allowed = {'.html', '.xml', '.txt', '.css', '.svg', '.png', '.jpg', '.webp', '.woff2'}
    for name in files:
        if PurePosixPath(name).suffix.lower() not in allowed:
            raise ValueError(f'Unexpected published file: {name}')
    if 'writing/index.xml' not in files:
        raise ValueError('Missing writing RSS feed.')
    feed = ET.fromstring(files['writing/index.xml'])
    links = [item.findtext('link') for item in feed.findall('./channel/item')]
    if len(links) != len(published) or set(links) != {urljoin(base, f'writing/{s}/') for s in published}:
        raise ValueError('RSS entries differ from published writing.')


def check_site(output: Path, base: str, published: set[str]) -> None:
    """Read a built directory, then apply the pure snapshot validation contract."""
    output = output.resolve()
    files = {}
    for path in output.rglob('*'):
        if path.is_file():
            if not path.resolve().is_relative_to(output):
                raise ValueError(f'Published file escapes output: {path.relative_to(output)}')
            files[path.relative_to(output).as_posix()] = path.read_bytes()
    check_site_snapshot(files, base, published)


def build(base: str) -> Path:
    base_url(base)
    version = subprocess.run(['hugo', 'version'], check=True, capture_output=True, text=True).stdout
    pattern = r'^hugo v' + re.escape(HUGO_VERSION) + r'(?:-[0-9a-f]{40})?(?:\+|\s)'
    if not re.search(pattern, version):
        raise ValueError(f'Use Hugo {HUGO_VERSION}; found {version.strip()}')
    destination = ROOT / '.cache/site/public'
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='dolittle-site-', dir=destination.parent) as temp:
        scratch = Path(temp)
        project, output = scratch / 'project', scratch / 'output'
        project.mkdir()
        published = stage(project)
        subprocess.run(['hugo', '--source', str(project), '--destination', str(output),
                        '--baseURL', base, '--panicOnWarning'], check=True)
        check_site(output, base, published)
        previous = scratch / 'previous'
        if destination.exists():
            destination.rename(previous)
        try:
            output.rename(destination)
        except OSError:
            if previous.exists():
                previous.rename(destination)
            raise
    print(f'Built and verified {len(published)} articles, Home, Writing, About, and 404: {destination}', flush=True)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['build', 'serve'])
    parser.add_argument('--base-url', help='Override the production baseURL from web/hugo.toml.')
    parser.add_argument('--port', type=int, default=1315)
    args = parser.parse_args()
    try:
        if not 1 <= args.port <= 65535:
            raise ValueError('Port must be between 1 and 65535.')
        configured_base = tomllib.loads(source_file('web/hugo.toml').read_text())['baseURL']
        base = f'http://127.0.0.1:{args.port}/' if args.command == 'serve' else (args.base_url or configured_base)
        output = build(base)
        if args.command == 'serve':
            print(f'Preview: {base} (Ctrl-C to stop; rerun after source edits)', flush=True)
            handler = partial(SimpleHTTPRequestHandler, directory=str(output))
            with ThreadingHTTPServer(('127.0.0.1', args.port), handler) as server:
                server.serve_forever()
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        parser.exit(1, f'Site build failed: {error}\n')
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
