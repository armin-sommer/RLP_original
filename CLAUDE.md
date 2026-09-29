# rp1

## Style

- **Comment only what the code/file itself cannot say** - a measurement, a paper's finding, a consequence
  that bites later. Never a restatement of the line below. This binds hardest in the config yamls,
  where the `defaults:` list and the field names already say what a config selects.
- **Describe the code as it is, never how it got there.** Nothing justifies a line against a past
  version, mentions a fix, or says what something used to do. A *dependency's* API history is fair game;
  it warns the reader off what its own examples still show.
- **Keep docstrings concise.** What the thing is, and what the caller has to know. A fact about one
  line is a comment on that line, not a paragraph at the top.
- **Never hard-code a reference or a figure that goes stale** - a directory tree in a markdown
  file, a line number, a path in prose, or a quantity worked out by hand from config values: a
  parameter count, a checkpoint size, a memory footprint. Nothing checks them and they are wrong by
  the next commit. Name the symbol and let the reader search for it; let a number that follows from
  the config be read off a run.
- **Write modular, maintainable code.** One responsibility per module, dependencies pointing one
  way, behaviour chosen by config rather than by editing a hardcoded target. Restructure rather
  than add a flag to work around the existing design.
- **`configs/` and `tests/` mirror `src/rp1/`.** Both are subsets - not every package is
  configurable or needs its own test file - but nothing sits at a path that does not correspond.
- **A default lives in one place.** If a yaml sets a value, the Python signature does not default
  it too. The two drift silently, and the yaml is what actually runs.
