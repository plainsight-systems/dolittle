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

Use level-two headings for sections and relative Hugo links between articles,
for example `{{< relref "/writing/two-views-of-a-thinking-model/" >}}`.
The site provides the title, byline, date, reading time, and contents navigation.

Separate measured results from plans and interpretations. Attribute reference
runs to the reference implementation. Neither the thinking text nor the NLA
interpretation is ground truth for the other. Never put machine access details
or private planning notes in public prose.

Build and preview instructions are in [`../../web/README.md`](../../web/README.md).
