"""One publication-build integration and isolated in-memory site guard units.

The integration uses a disposable repository and real Hugo build. Unit tests
pass byte snapshots directly to the same validator and never create files or
launch processes. Each negative case checks a publication contract.
"""
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('dolittle_site', ROOT / 'scripts/site.py')
site = importlib.util.module_from_spec(spec)
spec.loader.exec_module(site)


class PublicationIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        for folder in ('web', 'docs/writeups', 'docs/assets'):
            shutil.copytree(ROOT / folder, cls.root / folder)
        (cls.root / 'brief.md').write_text('PRIVATE_BRIEF_CANARY')
        (cls.root / 'docs/writeups/unlisted.md').write_text('UNLISTED_CANARY')
        manifest = cls.root / 'web/publications.json'
        data = json.loads(manifest.read_text())
        for slug, fields in {
            'draft-check': 'draft = true',
            'future-check': 'date = "2999-01-01T00:00:00Z"',
            'milestone-check': 'article_type = "milestone"\nmilestone = 1',
        }.items():
            meta = {'title': slug, 'description': 'Fixture article',
                    'date': '2020-01-01T00:00:00Z', 'article_type': 'exploration'}
            import tomllib
            meta.update(tomllib.loads(fields))
            header = '\n'.join(f'{k} = {json.dumps(v)}' for k, v in meta.items())
            (cls.root / f'docs/writeups/{slug}.md').write_text('+++\n' + header + '\n+++\nFixture body.')
            data['articles'].append(slug)
        manifest.write_text(json.dumps(data))
        cls.base = 'https://example.test/dolittle/'
        with patch.object(site, 'ROOT', cls.root):
            cls.output = site.build(cls.base)
        cls.published = set(data['articles']) - {'draft-check', 'future-check'}

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_publication_boundary_and_subpath(self):
        site.check_site(self.output, self.base, self.published)
        html = '\n'.join(p.read_text() for p in self.output.rglob('*.html'))
        for excluded in ('PRIVATE_BRIEF_CANARY', 'UNLISTED_CANARY', 'draft-check', 'future-check'):
            self.assertNotIn(excluded, html)
        self.assertIn('Milestone 1', html)
        self.assertIn('/dolittle/writing/milestone-check/', html)


class SiteValidationUnitTests(unittest.TestCase):
    base = 'https://example.test/dolittle/'
    published = {'example'}

    def setUp(self):
        self.files = {}
        names = ('index.html', 'writing/index.html', 'about/index.html', '404.html',
                 'writing/example/index.html')
        for name in names:
            page_url = self.base + name.removesuffix('index.html')
            metadata = {
                'og:title': 'Example', 'og:description': 'Description',
                'og:type': 'website', 'og:site_name': 'Dolittle',
                'og:url': page_url, 'og:image': self.base + 'share.jpg',
                'og:image:alt': 'Artwork', 'og:image:type': 'image/jpeg',
                'og:image:width': '1200', 'og:image:height': '400',
                'twitter:card': 'summary_large_image', 'twitter:title': 'Example',
                'twitter:description': 'Description', 'twitter:image': self.base + 'share.jpg',
                'twitter:image:alt': 'Artwork',
            }
            html = '<title>Example</title><main><h1 id="title">Example</h1></main>'
            html += f'<link rel="canonical" href="{page_url}">'
            html += ''.join(f'<meta property="{key}" content="{value}">'
                            for key, value in metadata.items())
            html += '<a href="' + self.base + 'writing/example/#title">Example</a>'
            html += '<link rel="stylesheet" href="' + self.base + 'style.css">'
            self.files[name] = html.encode()
        self.files['share.jpg'] = b'\xff\xd8fixture'
        self.files['style.css'] = b'@font-face{src:url(font.woff2)}'
        self.files['font.woff2'] = b'font fixture'
        self.files['writing/index.xml'] = (
            '<rss><channel><item><link>' + self.base + 'writing/example/'
            '</link></item></channel></rss>').encode()

    def validate(self):
        site.check_site_snapshot(self.files, self.base, self.published)

    def append_home(self, fragment):
        self.files['index.html'] += fragment.encode()

    def test_valid_snapshot_and_directory_links(self):
        before = dict(self.files)
        self.validate()
        self.assertEqual(self.files, before)

    def test_broken_anchor_is_rejected(self):
        self.append_home('<a href="#missing-anchor">Broken</a>')
        with self.assertRaises(ValueError):
            self.validate()

    def test_missing_responsive_asset_is_rejected(self):
        self.append_home('<img alt="" srcset="missing.webp 800w">')
        with self.assertRaises(ValueError):
            self.validate()

    def test_stale_share_identity_is_rejected(self):
        self.files['index.html'] = self.files['index.html'].replace(
            ('property="og:url" content="' + self.base).encode(),
            b'property="og:url" content="http://localhost:1315/')
        with self.assertRaises(ValueError):
            self.validate()

    def test_missing_share_image_is_rejected(self):
        del self.files['share.jpg']
        with self.assertRaises(ValueError):
            self.validate()

    def test_non_jpeg_share_image_is_rejected(self):
        self.files['share.jpg'] = b'not a JPEG'
        with self.assertRaises(ValueError):
            self.validate()

    def test_unexpected_public_document_is_rejected(self):
        self.files['brief.md'] = b'Unintended document'
        with self.assertRaises(ValueError):
            self.validate()

    def test_missing_font_is_rejected(self):
        self.files['style.css'] += b'@font-face{src:url(missing.woff2)}'
        with self.assertRaises(ValueError):
            self.validate()

    def test_unexpected_page_is_rejected(self):
        self.files['unexpected/index.html'] = self.files['index.html']
        with self.assertRaises(ValueError):
            self.validate()

    def test_encoded_output_escape_is_rejected(self):
        self.append_home('<a href="%2e%2e/private.txt">Escapes</a>')
        with self.assertRaises(ValueError):
            self.validate()

    def test_site_base_escape_is_rejected(self):
        self.append_home('<a href="/private.txt">Escapes</a>')
        with self.assertRaises(ValueError):
            self.validate()

    def test_feed_membership_must_match_publication_set(self):
        self.files['writing/index.xml'] = b'<rss><channel/></rss>'
        with self.assertRaises(ValueError):
            self.validate()


if __name__ == '__main__':
    unittest.main()
