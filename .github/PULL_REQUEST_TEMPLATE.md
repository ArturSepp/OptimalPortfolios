## What changed

Describe the problem and the smallest coherent change that solves it.

## Verification

List the exact commands run and their results.

## Checklist

- [ ] Tests cover the changed public behavior or defect.
- [ ] Backtest code uses no future observations and applies weights over the next return period.
- [ ] Return, annualisation, missing-data, and optimisation conventions remain explicit.
- [ ] No credentials, private/licensed data, local paths, generated outputs, or agent reports are included.
- [ ] `uv run --no-sync pytest` and the relevant static/docs checks pass.
- [ ] User-visible changes are documented in `CHANGELOG.md` and relevant docs.
- [ ] New runtime dependencies or public-signature changes are called out explicitly.
