# Verification

Verification will cover independent numerical comparisons of the target and
NLA computation, GPU execution on the 3090, model loading, and the browser/API
interaction. GPU tests and ordinary build checks will have separate entry
points. The GPU path must also run with forced PTX compilation.

Fixtures should be small and attributable. Full model weights belong in the
ignored cache, with pinned revisions and checksums. Tests arrive with the
implementation they verify. Engine verification remains upcoming.

The independent site checks run with Python 3.11+ and Hugo 0.154.2:

```sh
python3 -m unittest discover -s tests -p 'test_site.py' -v
```

They build a disposable site under a deployment subpath, check draft/future
exclusion and milestone rendering, and inject broken links, assets, and an
unintended document to verify the publication guards.
