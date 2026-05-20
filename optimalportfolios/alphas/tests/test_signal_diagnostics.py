"""
Tests for optimalportfolios.alphas.signal_diagnostics.

Exercises the AlphasData integration, the per-component sweep, the
comparison aggregation, and the plotting functions. Numerical
correctness of the underlying regression is covered in
qis.perfstats.tests.test_signal_diagnostics.
"""
# built-in
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import qis

# optimalportfolios
from optimalportfolios.alphas.alpha_data import AlphasData
from optimalportfolios.alphas.signal_diagnostics import (
    run_signal_diagnostics,
    run_signal_diagnostics_per_component,
    plot_signal_diagnostics,
    plot_signal_diagnostics_per_component,
    compare_signal_diagnostics,
)


def _make_synthetic_alphas_data(
        n_monthly: int = 10, n_quarterly: int = 3,
        n_years: int = 8, beta_true: float = 0.30, seed: int = 11,
) -> tuple:
    """Build a returns dict + AlphasData with two populated score fields."""
    rng = np.random.default_rng(seed)
    n_months = n_years * 12

    me_dates = pd.date_range('2012-01-31', periods=n_months, freq='ME')
    qe_dates = pd.date_range('2012-03-31', periods=n_years * 4, freq='QE')

    me_assets = [f"M{i:02d}" for i in range(n_monthly)]
    qe_assets = [f"Q{i:02d}" for i in range(n_quarterly)]
    all_assets = me_assets + qe_assets

    # Monthly signal panel (for all assets)
    z = rng.normal(0, 0.5, size=(n_months, len(all_assets))).clip(-1, 1)
    signal_panel = pd.DataFrame(z, index=me_dates, columns=all_assets)

    # Monthly returns: predictable via lagged signal for monthly assets
    z_lag = np.vstack([np.zeros((1, n_monthly)), z[:-1, :n_monthly]])
    f = rng.normal(0.005, 0.04, size=n_months)
    idio = rng.normal(0, 0.05, size=(n_months, n_monthly))
    r_m = f[:, None] + beta_true * 0.05 * z_lag + idio
    me_returns = pd.DataFrame(r_m, index=me_dates, columns=me_assets)

    # Quarterly returns: predictable at quarterly cadence
    z_q = signal_panel.loc[qe_dates, qe_assets].to_numpy()
    z_q_lag = np.vstack([np.zeros((1, n_quarterly)), z_q[:-1, :]])
    f_q = rng.normal(0.015, 0.07, size=len(qe_dates))
    idio_q = rng.normal(0, 0.08, size=(len(qe_dates), n_quarterly))
    r_q = f_q[:, None] + beta_true * 0.08 * z_q_lag + idio_q
    qe_returns = pd.DataFrame(r_q, index=qe_dates, columns=qe_assets)

    asset_returns_dict = {'ME': me_returns, 'QE': qe_returns}

    # A noise score panel for component sweep tests
    noise = pd.DataFrame(
        rng.normal(0, 0.5, size=(n_months, len(all_assets))).clip(-1, 1),
        index=me_dates, columns=all_assets,
    )

    ad = AlphasData(
        alpha_scores=signal_panel,
        momentum_score=signal_panel,
        beta_score=noise,
    )

    half = len(all_assets) // 2
    group_data = pd.Series(
        ['A'] * half + ['B'] * (len(all_assets) - half), index=all_assets,
    )
    return asset_returns_dict, ad, group_data


# ───────────────────────────────────────────────────────────────────────────────
# Compute
# ───────────────────────────────────────────────────────────────────────────────


class TestRunSignalDiagnostics:
    """run_signal_diagnostics — pure compute."""

    def test_accepts_alphas_data(self):
        ard, ad, gd = _make_synthetic_alphas_data()
        result = run_signal_diagnostics(asset_returns_dict=ard, signal=ad,
                                        group_data=gd, horizons=[1, 3])
        assert isinstance(result, qis.SignalDiagnosticsResult)
        assert list(result.pooled_universe.index) == ['1', '3']

    def test_default_horizons_include_2(self):
        """Default horizons should be (1, 2, 3, 6) — call without horizons kwarg."""
        ard, ad, gd = _make_synthetic_alphas_data()
        result = run_signal_diagnostics(asset_returns_dict=ard, signal=ad,
                                        group_data=gd)
        assert list(result.pooled_universe.index) == ['1', '2', '3', '6']

    def test_accepts_dataframe(self):
        ard, ad, _ = _make_synthetic_alphas_data()
        result = run_signal_diagnostics(asset_returns_dict=ard,
                                        signal=ad.alpha_scores, horizons=[1])
        assert isinstance(result, qis.SignalDiagnosticsResult)

    def test_signal_attribute_selection(self):
        ard, ad, gd = _make_synthetic_alphas_data()
        r1 = run_signal_diagnostics(asset_returns_dict=ard, signal=ad,
                                    signal_attribute='alpha_scores',
                                    group_data=gd, horizons=[1])
        r2 = run_signal_diagnostics(asset_returns_dict=ard, signal=ad,
                                    signal_attribute='beta_score',
                                    group_data=gd, horizons=[1])
        assert np.isfinite(r1.pooled_universe.loc['1', 'beta'])
        assert np.isfinite(r2.pooled_universe.loc['1', 'beta'])

    def test_raises_on_missing_attribute(self):
        ard, ad, _ = _make_synthetic_alphas_data()
        with pytest.raises(AttributeError):
            run_signal_diagnostics(asset_returns_dict=ard, signal=ad,
                                   signal_attribute='not_a_field',
                                   horizons=[1])

    def test_raises_when_signal_attribute_is_none(self):
        ard, ad, _ = _make_synthetic_alphas_data()
        with pytest.raises(ValueError, match='is None'):
            run_signal_diagnostics(asset_returns_dict=ard, signal=ad,
                                   signal_attribute='residual_momentum_score',
                                   horizons=[1])


class TestRunSignalDiagnosticsPerComponent:
    """run_signal_diagnostics_per_component — compute sweep."""

    def test_runs_only_populated_components(self):
        ard, ad, gd = _make_synthetic_alphas_data()
        results = run_signal_diagnostics_per_component(
            asset_returns_dict=ard, alphas_data=ad,
            group_data=gd, horizons=[1, 3],
        )
        assert set(results.keys()) == {'alpha_scores', 'momentum_score',
                                       'beta_score'}

    def test_components_filter(self):
        ard, ad, gd = _make_synthetic_alphas_data()
        results = run_signal_diagnostics_per_component(
            asset_returns_dict=ard, alphas_data=ad, group_data=gd,
            horizons=[1], components=['alpha_scores'],
        )
        assert list(results.keys()) == ['alpha_scores']


class TestCompareSignalDiagnostics:
    """compare_signal_diagnostics — comparison table."""

    def test_returns_empty_for_empty_dict(self):
        assert compare_signal_diagnostics({}).empty

    def test_aggregates_pooled_rows_multi_horizon(self):
        ard, ad, gd = _make_synthetic_alphas_data()
        results = run_signal_diagnostics_per_component(
            asset_returns_dict=ard, alphas_data=ad,
            group_data=gd, horizons=[1, 3],
        )
        out = compare_signal_diagnostics(results)
        assert out.index.names == ['signal', 'horizon']
        assert len(out) == 6   # 3 signals × 2 horizons

    def test_aggregates_pooled_rows_single_horizon(self):
        ard, ad, gd = _make_synthetic_alphas_data()
        results = run_signal_diagnostics_per_component(
            asset_returns_dict=ard, alphas_data=ad,
            group_data=gd, horizons=[1, 3],
        )
        out = compare_signal_diagnostics(results, horizon='3')
        assert out.index.name == 'signal'
        assert len(out) == 3
        assert 'beta' in out.columns


# ───────────────────────────────────────────────────────────────────────────────
# Plot
# ───────────────────────────────────────────────────────────────────────────────


class TestPlotSignalDiagnostics:
    """plot_signal_diagnostics — returns a Figure, no I/O."""

    def test_returns_figure_for_alphas_data(self):
        ard, ad, gd = _make_synthetic_alphas_data()
        fig = plot_signal_diagnostics(asset_returns_dict=ard, signal=ad,
                                      group_data=gd, horizons=[1, 3])
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_returns_figure_for_dataframe(self):
        ard, ad, _ = _make_synthetic_alphas_data()
        fig = plot_signal_diagnostics(asset_returns_dict=ard,
                                      signal=ad.alpha_scores, horizons=[1])
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_custom_title_is_respected(self):
        ard, ad, gd = _make_synthetic_alphas_data()
        fig = plot_signal_diagnostics(asset_returns_dict=ard, signal=ad,
                                      group_data=gd, horizons=[1],
                                      title='Custom title for this test')
        assert fig._suptitle.get_text() == 'Custom title for this test'
        plt.close(fig)

    def test_auto_title_includes_signal_label(self):
        ard, ad, gd = _make_synthetic_alphas_data()
        fig = plot_signal_diagnostics(asset_returns_dict=ard, signal=ad,
                                      group_data=gd, horizons=[1])
        assert "'alpha_scores'" in fig._suptitle.get_text()
        plt.close(fig)


class TestPlotSignalDiagnosticsPerComponent:
    """plot_signal_diagnostics_per_component — dict of Figures."""

    def test_returns_dict_of_figures(self):
        ard, ad, gd = _make_synthetic_alphas_data()
        figs = plot_signal_diagnostics_per_component(
            asset_returns_dict=ard, alphas_data=ad,
            group_data=gd, horizons=[1, 3],
        )
        assert set(figs.keys()) == {'alpha_scores', 'momentum_score',
                                    'beta_score'}
        for fig in figs.values():
            assert isinstance(fig, plt.Figure)
            plt.close(fig)

    def test_components_filter_propagates(self):
        ard, ad, gd = _make_synthetic_alphas_data()
        figs = plot_signal_diagnostics_per_component(
            asset_returns_dict=ard, alphas_data=ad, group_data=gd,
            horizons=[1], components=['alpha_scores'],
        )
        assert list(figs.keys()) == ['alpha_scores']
        plt.close(figs['alpha_scores'])


class TestPlotSignalDiagnosticsBetaBoxplot:
    """plot_signal_diagnostics_beta_boxplot — per-asset β distribution."""

    def test_returns_figure(self):
        from optimalportfolios.alphas.signal_diagnostics import (
            plot_signal_diagnostics_beta_boxplot,
        )
        ard, ad, gd = _make_synthetic_alphas_data(n_years=10)
        fig = plot_signal_diagnostics_beta_boxplot(
            asset_returns_dict=ard, signal=ad, group_data=gd,
            horizons=[1, 2, 3, 6], min_obs_per_asset=12,
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_handles_no_qualifying_assets(self):
        from optimalportfolios.alphas.signal_diagnostics import (
            plot_signal_diagnostics_beta_boxplot,
        )
        ard, ad, gd = _make_synthetic_alphas_data(n_years=10)
        # Absurd min_obs filter → no assets survive, function should still
        # produce a Figure with an info message rather than raising.
        fig = plot_signal_diagnostics_beta_boxplot(
            asset_returns_dict=ard, signal=ad, group_data=gd,
            horizons=[1], min_obs_per_asset=999_999,
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_custom_title_respected(self):
        from optimalportfolios.alphas.signal_diagnostics import (
            plot_signal_diagnostics_beta_boxplot,
        )
        ard, ad, gd = _make_synthetic_alphas_data(n_years=10)
        fig = plot_signal_diagnostics_beta_boxplot(
            asset_returns_dict=ard, signal=ad, group_data=gd,
            horizons=[1, 2], title='Custom boxplot title',
        )
        # Title is on the axes, not the figure suptitle (single-panel).
        ax = fig.axes[0]
        assert ax.get_title() == 'Custom boxplot title'
        plt.close(fig)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])