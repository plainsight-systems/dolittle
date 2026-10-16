"""Exercise publication isolation, URL integrity and a real subpath build.

Use a disposable repository fixture; never mutate working articles or previews.
The build verifies generated output, while injected failures check that those
guards catch broken assets, anchors, share metadata and accidental publication.
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


class PublicationTests(unittest.TestCase):
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

    def test_broken_anchor_and_responsive_asset_are_rejected(self):
        home = self.output / 'index.html'
        original = home.read_text()
        try:
            for fragment, message in [
                ('<a href="#missing-anchor">Broken</a>', 'missing fragment'),
                ('<img alt="" srcset="missing.webp 800w">', 'Missing target'),
            ]:
                with self.subTest(fragment=fragment):
                    home.write_text(original + fragment)
                    with self.assertRaisesRegex(ValueError, message):
                        site.check_site(self.output, self.base, self.published)
        finally:
            home.write_text(original)

    def test_stale_share_identity_and_missing_image_are_rejected(self):
        home = self.output / 'index.html'
        original = home.read_text()
        page = site.Page(original)
        try:
            home.write_text(original.replace(
                'property="og:url" content="' + self.base,
                'property="og:url" content="http://localhost:1315/'))
            with self.assertRaisesRegex(ValueError, 'invalid social metadata'):
                site.check_site(self.output, self.base, self.published)
            home.write_text(original.replace(page.meta['og:image'], self.base + 'missing.jpg'))
            with self.assertRaisesRegex(ValueError, 'Missing target'):
                site.check_site(self.output, self.base, self.published)
        finally:
            home.write_text(original)

    def test_unexpected_public_document_is_rejected(self):
        leaked = self.output / 'brief.md'
        try:
            leaked.write_text('Unintended document')
            with self.assertRaisesRegex(ValueError, 'Unexpected published file'):
                site.check_site(self.output, self.base, self.published)
        finally:
            leaked.unlink()

    def test_missing_font_is_rejected(self):
        css = next(self.output.rglob('*.css'))
        original = css.read_text()
        try:
            css.write_text(original + '@font-face{src:url(missing.woff2)}')
            with self.assertRaisesRegex(ValueError, 'Missing target'):
                site.check_site(self.output, self.base, self.published)
        finally:
            css.write_text(original)


if __name__ == '__main__':
    unittest.main()
