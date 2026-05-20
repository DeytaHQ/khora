# Khora memory plugin for Hermes

Drop-in long-term memory for [Hermes](https://github.com/hermes-agent)
agents, backed by Khora.

## Install

```bash
pip install khora hermes-agent
cp -r examples/integrations/hermes/plugin "$HERMES_HOME/plugins/khora"
```

There is no `[hermes]` extra on khora. `hermes-agent==0.13.0`
exact-pins `requests==2.33.0`, which clashes with khora's
`requests>=2.33.1` floor (CVE-2026-25645); the extra was dropped
during Wave C. Install `hermes-agent` yourself and accept whichever
`requests` version your resolver picks, or fork the upstream pin.
See `docs/integrations/hermes.md` for the full posture.

Hermes scans `$HERMES_HOME/plugins/<name>/` on startup and calls
`register(ctx)` from `__init__.py` once per plugin. No further wiring
needed — `KhoraMemoryProvider` is registered against the running
context.

## Configuration

The plugin reads one optional environment variable:

- `KHORA_HERMES_KB_FACTORY` — import path of a zero-arg callable that
  returns a `Khora` instance, in `module.path:attr` form (e.g.
  `myapp.memory:build_kb`). When unset, the plugin calls
  `Khora.shared()`, which picks up the standard `KHORA_*` env vars
  (`KHORA_STORAGE__*`, `KHORA_LLM__*`, etc.) — see
  `docs/configuration.md`.

The process-wide `Khora.shared()` singleton is cached by config hash, so
multiple Hermes sessions in the same process share one backend pool.

## Troubleshooting

- **`ImportError: No module named 'hermes_agent'`** — install
  `hermes-agent` in the same venv as khora: `pip install hermes-agent`.
- **Resolver complains about `requests`** — `hermes-agent` exact-pins
  `requests==2.33.0`. Either pin `requests==2.33.0` in your project
  and accept the unpatched CVE, or fork `hermes-agent` to relax the
  pin.
- **Plugin not discovered on startup** — confirm `$HERMES_HOME` is set
  and the directory contains both `plugin.yaml` and `__init__.py`.
  Hermes logs the discovery scan at INFO; check for the `khora` entry.
- **`KHORA_HERMES_KB_FACTORY` import fails** — the value must be
  `package.module:attr`, where `attr` is a zero-arg callable. The
  callable is invoked once at plugin registration.

## See also

- `docs/integrations/hermes.md` — full quickstart and reference.
- `src/khora/integrations/hermes/` — adapter source.
