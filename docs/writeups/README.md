# Writing

Writing is Dolittle's main public thread. Explanations open up the machinery;
explorations follow questions, observations, and decisions; milestones record
completed capabilities. Articles do not need a milestone number or a fixed
reading order.

Create a Markdown file with a lowercase hyphenated name and TOML frontmatter:

```toml
+++
title = 'Article title'
description = 'A short description for the index and feed.'
date = '2026-10-16T09:00:00Z'
article_type = 'exploration'
draft = true
+++
```

Use `explanation`, `exploration`, or `milestone` for `article_type`. A milestone
also requires a positive integer `milestone` field. The date must include a
time zone. Add the filename without `.md` to `web/publications.json` when it
belongs in the site's publication list. Only listed, non-draft articles whose
date has arrived enter the generated site and RSS. Removing `draft = true`
prepares an article for the next build; it does not deploy the site.

When revising a published article, preserve `date` and add `lastmod` with a
time-zone-qualified timestamp. The article and index show its updated date.
Date later additions explicitly when they describe work after the original
publication, so current capabilities do not read as historical results.

Use level-two headings for sections and relative Hugo links between articles,
for example `{{< relref "/writing/two-views-of-a-thinking-model/" >}}`.
The site provides the title, byline, date, reading time, and contents navigation.

Walkthroughs should show the actual application at the steps being explained.
Keep source screenshots in `web/assets/walkthrough/` and use the `screenshot`
shortcode with `src`, `alt`, and `caption`. It generates responsive WebP images
and links to the full-size capture. Captions identify the controls and describe
the captured run; do not present a particular sample as an expected result.

Separate measured results from plans and interpretations. Attribute reference
runs to the reference implementation. Neither the thinking text nor the NLA
interpretation is ground truth for the other. Never put machine access details
or private planning notes in public prose.

Build and preview instructions are in [`../../web/README.md`](../../web/README.md).
