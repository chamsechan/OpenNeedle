# README artwork

The README uses a self-contained SVG with light and dark palettes. Subtle packet
motion illustrates one packed matrix–vector projection; it does not represent
measured latency. Throughput bars are static, share a zero origin, and use the
recorded expanded-suite values in [`performance_f4f9b38.json`](../../reports/performance_f4f9b38.json).
The chart shows optimized and baseline OpenNeedle with SDOT + INT8 KV, both using
4 threads and nine measurements per case. The official result is from an earlier
five-repeat run and reports its own TPS. The 1.6× callout compares the two native
revisions, which were interleaved; it does not compare against PyTorch or official.
See [methodology](../backend-comparison.md) for the different timing boundaries,
quantization tradeoffs and the 16 distinct requests.

- [Light animated SVG](openeedle-hero-light.svg)
- [Dark animated SVG](openeedle-hero-dark.svg)
- [Static SVG](openeedle-hero-static.svg), suitable for slides and documents
- [Dark static SVG](openeedle-hero-static-dark.svg)

Regenerate from the repository root with Python's standard library:

```bash
python scripts/render_readme_assets.py
```

This reads the saved report; it does not run a benchmark. Model size and architecture
labels are editorial annotations in the generator and should be reviewed if the
model changes. Each SVG embeds the report's SHA-256 for traceability.

GitHub supports [`<picture>` and relative image paths](https://docs.github.com/en/get-started/writing-on-github/getting-started-with-writing-and-formatting-syntax/basic-writing-and-formatting-syntax#images).
The images contain no JavaScript, external resources, or embedded HTML, following
the restrictions for [SVG used as an image](https://developer.mozilla.org/en-US/docs/Web/SVG/Guides/SVG_as_an_image).
The README's `<picture>` selects a static asset when
[`prefers-reduced-motion`](https://developer.mozilla.org/en-US/docs/Web/CSS/Reference/At-rules/@media/prefers-reduced-motion)
is enabled. This also works when a browser does not propagate that preference
into an SVG image. The animated files include the same CSS media query for
direct viewing. All labels and measurements remain visible when animation is
disabled or unsupported.
