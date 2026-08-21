# Escalation — manuscript-finalisation frozen inputs (resolved)

**Date:** 2026-08-20  
**Stage:** F0  
**Status:** RESOLVED by owner instruction on 2026-08-20.

The first F0 inventory found 56 of 65 inputs and escalated nine missing artifacts: six
signal NAV/weight files, two risk NAV files, and the U1 ME/36 delta-0.0866 adopted-cell
cache. The owner then instructed: "if data is missing you need to do re-run". This was
treated as a narrow amendment authorizing reconstruction of exactly those nine artifacts.

No search or specification change was made. Frozen elected signal grids were replayed;
U1 risk was replayed; U3 risk NAVs were rebuilt from its accepted seven-exclusion dated
weights and reproduced the frozen performance table within `6.279699e-15` against a
`1e-12` tolerance; and the adopted U1 ME/36 smoothing cell was rebuilt at exactly delta
0.0866 with 203/203 snapshots and 100% injected/fitted partition agreement.

The final deterministic inventory resolves 65/65 inputs exactly once, with zero missing
and zero ambiguous paths. Inventory SHA-256:
`3B2E76DD51998E6690C04A426987CDA583EAE512FCB01224FB12256C02D4589B`.

The report of record is `agents/2026-08-20_sol_F0_report.md`. This escalation is retained
as the audit trail for the owner-authorized exception and no longer blocks F1.
