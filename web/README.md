# Project website

Dolittle's public site follows Cradle's explanatory reading format, with
Serenity's milestone writing available alongside explanations and explorations.
It is useful without running the GPU engine. The live inference interface
belongs in `app/`.

## Build and preview

Requirements: Python 3.11+ and Hugo **0.154.2**. No Python packages, JavaScript
runtime, model downloads, or GPU are needed.

From the repository root:

```sh
python3 scripts/site.py serve
python3 -m unittest discover -s tests -p 'test_site.py' -v
```

The preview is at <http://127.0.0.1:1315/>. Stop with Ctrl-C and rerun after
editing source files. `--port 1316` selects a different loopback port.

To build for https://dolittle.plainsight-systems.com/:

```sh
python3 scripts/site.py build
```

Verified output goes to `.cache/site/public/`. The build checks the exact
public page set, HTML structure, local links and fragments, responsive image
sources, CSS assets, and RSS membership before replacing the last good output.
The production domain is configured in `hugo.toml`. GitHub Pages uses the
`.github/workflows/pages.yml` workflow: pull requests build and validate; pushes
to `main` also publish the generated site. Hugo and Actions are pinned, and
only the deploy job receives Pages write permission. `--base-url https://example.com/dolittle/` overrides it when needed,
including a project subpath. `serve` always uses its loopback URL. The commands above
build locally; publication happens through the workflow.

Every page includes a canonical URL, Open Graph metadata, and a large-image
Twitter card with its own title and description. The existing full illustration
is generated as a 1200 × 400 JPEG for sharing, with image dimensions and alt
text. The build verifies the metadata identity and image file. Live crawler
previews can be checked once the domain serves the site. The GitHub navigation
link uses `params.repository`, set to https://github.com/plainsight-systems/dolittle.

## Content

- `content/`: home, Writing index, and About copy.
- `layouts/`: shared shell, editorial layouts, RSS, and markup renderers.
- `assets/`: CSS; the builder adds the original project illustration.
- `static/`: self-hosted IBM Plex fonts, their OFL license, and favicon.
- `data/observation.json`: attributable excerpts from the reference run.
- `publications.json`: explicit list of article slugs to include.
- `../docs/writeups/`: article source and authoring instructions.

Only the named write-ups and artwork are staged from `docs/`. Private notes,
process documents, raw research output, and the source brief are not mounted.
Everything under the site's own content/static directories is public input.
Do not place private material there. Drafts and future-dated articles are
excluded from staging, indexes, and RSS.

The site has no client-side JavaScript. It uses native contents disclosures,
responsive tables, visible keyboard focus, and a single reading column on
small screens. See [the site architecture](../docs/architecture/project-site.md).
