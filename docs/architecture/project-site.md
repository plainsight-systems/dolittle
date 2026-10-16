# Project site

The public site is the current implementation priority, before engine
architecture. It explains the questions, mechanisms, and observations behind
Dolittle. Andrew chose Cradle as the closer reference, with Serenity's
milestone writing retained as one kind of article.

## Structure and responsibilities

- Home introduces the project with its existing artwork and a route into the
  writing. An actual, labeled reference excerpt introduces the two views.
- Writing is a chronological index of explanations, explorations, and
  milestones. It does not impose a numbered curriculum or manufacture future
  milestone pages. Articles have dates, a type, descriptions, and contents.
- About explains the motivation, implementation intent, and current status.
- A writing RSS feed uses the same published article set as the index.

The site is static Hugo 0.154.2, following the sibling sites. Python 3.11+
stages and checks its inputs. No engine, CUDA, JavaScript framework, remote
fonts, or model downloads are needed to build or read it.

## Change axes and file ownership

| Change | Owner |
| --- | --- |
| Public article prose and metadata | `docs/writeups/*.md` |
| Explicit choice of articles to expose | `web/publications.json` |
| Home/About/index copy | `web/content/`, homepage layout |
| Shared reading/navigation structure | `web/layouts/` |
| Visual identity and responsive layout | `web/assets/site.css`, existing artwork |
| A selected reference excerpt | `web/data/observation.json` with source provenance |
| Staging, publication boundaries, verification, local serving | `scripts/site.py` |
| Boundary and generated-output regression checks | `tests/test_site.py` |

## Publication contract

The builder stages only the web-owned files, explicitly listed articles, and
the named Dolittle illustration. No mount exposes docs, process records,
research outputs, cached models, or the private brief. Article metadata lives
with the prose; the publication list only identifies approved source names.
Drafts and future articles stay out of the generated pages, index, and feed.
The normal build never enables Hugo draft/future flags.

Generated local links, anchors, responsive image sources, CSS assets, page
landmarks, RSS entries, and the public page set are checked before the last
good output is replaced. Serve only generated output over loopback. The base
URL defaults to `https://dolittle.plainsight-systems.com/` for builds and is
configurable for a project subpath. Serving uses loopback. GitHub Pages builds
and publishes through `.github/workflows/pages.yml`. Pull requests validate
without deploying; pushes to main publish only `.cache/site/public`. The
pinned Hugo installer verifies the release archive checksum. Share metadata uses
page-specific titles/descriptions and an absolute JPEG image URL; the builder
checks those against the selected base URL. The generated image preserves the
full illustration at 1200 × 400, with dimensions and alternative text in the
Open Graph tags. The contract follows https://ogp.me/.

## Visual and content contract

Use Cradle's family typography and navy/ivory framing with Dolittle's art.
Long articles have a contents rail on desktop and a native disclosure on
small screens. Reading and navigation work without JavaScript. Explicitly
label illustrative artwork and reference excerpts; never present the static
site as the live inference application. Milestone entries use the same reading
layout and an explicit kind rather than a separate progress dashboard.
