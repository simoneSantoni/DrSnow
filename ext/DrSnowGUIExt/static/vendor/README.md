Vendored third-party assets served by the DrSnow GUI (no CDN is contacted at runtime).

| File | Source | Version | License |
|------|--------|---------|---------|
| `plotly-basic.min.js` | npm `plotly.js-basic-dist-min` | 3.7.0 | MIT (`plotly-LICENSE.txt`) |

`plotly-basic.min.js` is loaded with Subresource Integrity: its SHA-384 is in
`../index.html` and checked by `test/gui/runtests.jl`. To upgrade, replace the file
and update the `integrity` attribute and the version above.
