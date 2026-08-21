"""
the ``UniverseData`` container: validation, accessors, selection, round-tripping, transforms.

``optimalportfolios.universe.run_local.universe_data_run`` is a ``run_local`` diagnostic needing
the author's data, so the container itself had no collected tests. It is a frozen dataclass that
validates on construction, and *the validation is the point*: every check exists because a
misaligned universe otherwise fails much later, inside an optimiser, as a shape error with no clue
which asset caused it. So each check gets a case that trips it.

Everything here is built in the test. The save/load round trip writes into pytest's ``tmp_path``
rather than the repository or the author's data directory, so the suite stays offline and
leaves nothing behind.
"""
# packages
from enum import Enum
import numpy as np
import pandas as pd
import pytest
import qis
# optimalportfolios
from optimalportfolios.universe.universe_data import MetadataField, UniverseData
from optimalportfolios.universe.universe_transforms import (
    copy_universe_data_with_unsmoothed_prices)

SEED = 20260810
ASSETS = ['equity_us', 'equity_eu', 'bond_govt', 'private_equity']
ASSET_CLASSES = ['Equities', 'Equities', 'Bonds', 'Equities']


def make_prices(n_days: int = 900) -> pd.DataFrame:
    """A seeded daily price panel over ASSETS."""
    rng = np.random.default_rng(SEED)
    dates = pd.date_range('2021-01-01', periods=n_days, freq='B')
    returns = rng.normal(0.0004, 0.010, size=(n_days, len(ASSETS)))
    # the private-equity leg is deliberately smoothed, so unsmoothing has something to undo
    returns[:, 3] = pd.Series(returns[:, 3]).ewm(span=12).mean().to_numpy()
    return pd.DataFrame(100.0 * np.exp(np.cumsum(returns, axis=0)), index=dates, columns=ASSETS)


def make_metadata() -> pd.DataFrame:
    """Metadata carrying exactly the columns ``MetadataField`` requires."""
    return pd.DataFrame({MetadataField.NAME.value: ['US Equity', 'EU Equity',
                                                    'Govt Bonds', 'Private Equity'],
                         MetadataField.ASSET_CLASS.value: ASSET_CLASSES,
                         MetadataField.CURRENCY.value: ['USD', 'EUR', 'USD', 'USD']},
                        index=ASSETS)


def make_group_loadings() -> pd.DataFrame:
    """A binary asset-to-group mapping indexed by asset."""
    return pd.DataFrame({'Growth': [1.0, 1.0, 0.0, 1.0], 'Defensive': [0.0, 0.0, 1.0, 0.0]},
                        index=ASSETS)


def make_universe(**overrides) -> UniverseData:
    """A valid universe, with any field replaced by a keyword override."""
    kwargs = dict(prices=make_prices(), metadata=make_metadata(),
                  group_loadings_level1=make_group_loadings())
    kwargs.update(overrides)
    return UniverseData(**kwargs)


# --------------------------------------------------------------------------- #
# validation
# --------------------------------------------------------------------------- #
def test_a_well_formed_universe_constructs() -> None:
    """the happy path validates on construction and exposes its assets"""
    universe = make_universe()
    assert universe.assets == ASSETS
    assert universe.n_assets == len(ASSETS)


def test_prices_and_metadata_must_describe_the_same_assets() -> None:
    """an asset priced but not described would reach the optimiser as a shape error"""
    with pytest.raises(ValueError, match='Asset mismatch'):
        make_universe(metadata=make_metadata().drop(index='bond_govt'))


def test_metadata_must_carry_every_required_field() -> None:
    """the metadata_fields enum states the contract, so a missing column is rejected"""
    with pytest.raises(ValueError, match='missing required columns'):
        make_universe(metadata=make_metadata().drop(columns=[MetadataField.CURRENCY.value]))


def test_duplicate_price_columns_are_rejected() -> None:
    """a duplicated asset silently doubles an allocation, so it is caught up front"""
    prices = make_prices()
    duplicated = pd.concat([prices, prices[['equity_us']]], axis=1, sort=True)
    with pytest.raises(ValueError, match='Duplicate asset names in prices'):
        UniverseData(prices=duplicated, metadata=make_metadata())


def test_duplicate_metadata_rows_are_rejected() -> None:
    """the same check applies on the metadata side, where prices alone would not catch it

    The price columns stay unique here on purpose: a set comparison cannot see the repeated
    metadata row, so alignment passes and only the duplicate check stands between this and a
    silently doubled asset.
    """
    metadata = pd.concat([make_metadata(), make_metadata().loc[['equity_us']]], axis=0,
                         sort=False)
    with pytest.raises(ValueError, match='Duplicate asset names in metadata'):
        UniverseData(prices=make_prices(), metadata=metadata)


def test_nulls_in_a_required_metadata_column_are_rejected() -> None:
    """a missing currency or asset class breaks grouping later, so it fails here"""
    metadata = make_metadata()
    metadata.loc['bond_govt', MetadataField.CURRENCY.value] = None
    with pytest.raises(ValueError, match='Null values in required metadata columns'):
        make_universe(metadata=metadata)


def test_group_loadings_must_be_indexed_by_the_priced_assets() -> None:
    """a loadings table on a different universe would misattribute every group"""
    with pytest.raises(ValueError, match="group_loadings_level1 index doesn't match"):
        make_universe(group_loadings_level1=make_group_loadings().drop(index='equity_eu'))


def test_group_loadings_level2_is_validated_too() -> None:
    """both levels are checked, not just the first"""
    with pytest.raises(ValueError, match="group_loadings_level2 index doesn't match"):
        make_universe(group_loadings_level2=make_group_loadings().drop(index='equity_eu'))


def test_validation_can_be_switched_off_for_a_known_partial_universe() -> None:
    """validate_on_init exists for the loader, which validates after assembling"""
    universe = UniverseData(prices=make_prices(),
                            metadata=make_metadata().drop(index='bond_govt'),
                            validate_on_init=False)
    assert universe.n_assets == len(ASSETS)
    # the check is deferred, not removed
    with pytest.raises(ValueError, match='Asset mismatch'):
        universe.validate()


def test_a_custom_metadata_enum_defines_its_own_contract() -> None:
    """metadata_fields is replaceable, so a universe can require different columns"""
    class SparseField(str, Enum):
        """Only a name is required for this universe."""
        NAME = 'name'

    universe = UniverseData(prices=make_prices(),
                            metadata=make_metadata()[[MetadataField.NAME.value]],
                            metadata_fields=SparseField)
    assert universe.n_assets == len(ASSETS)


# --------------------------------------------------------------------------- #
# accessors
# --------------------------------------------------------------------------- #
def test_metadata_accessors_return_the_named_columns() -> None:
    """name, asset_class and currency are the three fields the base contract requires"""
    universe = make_universe()
    assert universe.name['equity_us'] == 'US Equity'
    assert list(universe.asset_class) == ASSET_CLASSES
    assert universe.currency['equity_eu'] == 'EUR'


def test_date_range_reports_the_first_and_last_price_date() -> None:
    """the range comes off the price index, not the metadata"""
    universe = make_universe()
    first, last = universe.date_range
    assert first == universe.prices.index[0]
    assert last == universe.prices.index[-1]


def test_hedge_ratio_flags_the_named_asset_classes() -> None:
    """the hedge ratio is one for assets in a hedged class and zero elsewhere"""
    ratio = make_universe().get_hedge_ratio(hedged_acs=['Bonds'])
    assert ratio['bond_govt'] == 1.0
    assert ratio['equity_us'] == 0.0
    assert set(ratio.index) == set(ASSETS)


def test_asset_returns_dict_keeps_the_first_observation_as_zero() -> None:
    """the container asks qis not to drop the first row, so panels stay aligned"""
    returns = make_universe().get_asset_returns_dict(returns_freqs='ME')
    assert isinstance(returns, dict)
    frame = next(iter(returns.values()))
    assert list(frame.columns) == ASSETS
    assert np.allclose(frame.iloc[0].to_numpy(), 0.0)


def test_asset_returns_dict_supports_log_returns() -> None:
    """the return convention is stated by the caller, never implied"""
    universe = make_universe()
    arithmetic = next(iter(universe.get_asset_returns_dict(is_log_returns=False).values()))
    logarithmic = next(iter(universe.get_asset_returns_dict(is_log_returns=True).values()))
    # log returns sit below arithmetic ones for positive moves; they are not the same panel
    assert not np.allclose(arithmetic.to_numpy(), logarithmic.to_numpy())


# --------------------------------------------------------------------------- #
# construction helpers
# --------------------------------------------------------------------------- #
def test_from_selection_subsets_every_table_together() -> None:
    """selecting assets slices prices, metadata and both loadings tables consistently"""
    universe = UniverseData.from_selection(
        prices=make_prices(), metadata=make_metadata(), assets=['equity_us', 'bond_govt'],
        group_loadings_level1=make_group_loadings())
    assert universe.assets == ['equity_us', 'bond_govt']
    assert list(universe.metadata.index) == ['equity_us', 'bond_govt']
    assert list(universe.group_loadings_level1.index) == ['equity_us', 'bond_govt']


def test_rename_index_switches_tickers_for_names_everywhere() -> None:
    """renaming has to move prices, metadata and loadings together or validation fails"""
    renamed = make_universe().rename_index()
    assert renamed.assets == ['US Equity', 'EU Equity', 'Govt Bonds', 'Private Equity']
    assert list(renamed.metadata.index) == renamed.assets
    assert list(renamed.group_loadings_level1.index) == renamed.assets


def test_rename_index_leaves_absent_loadings_absent() -> None:
    """the optional tables stay None rather than becoming empty frames"""
    renamed = UniverseData(prices=make_prices(), metadata=make_metadata()).rename_index()
    assert renamed.group_loadings_level1 is None
    assert renamed.group_loadings_level2 is None


def test_rename_index_moves_the_second_loadings_level_too() -> None:
    """level2 is renamed on the same map as level1, or its index stops matching the prices"""
    level2 = make_group_loadings().rename(columns={'Growth': 'Liquid', 'Defensive': 'Illiquid'})
    renamed = make_universe(group_loadings_level2=level2).rename_index()
    assert list(renamed.group_loadings_level2.index) == renamed.assets
    assert list(renamed.group_loadings_level2.columns) == ['Liquid', 'Illiquid']


# --------------------------------------------------------------------------- #
# persistence
# --------------------------------------------------------------------------- #
def test_save_and_load_round_trips_the_universe(tmp_path) -> None:
    """what is written is what comes back, including the group loadings"""
    universe = make_universe()
    local_path = f"{tmp_path}/"
    universe.save(file_name='test_universe', local_path=local_path)
    loaded = UniverseData.load(file_name='test_universe', local_path=local_path,
                               metadata_fields=MetadataField,
                               group_loadings_keys=['group_loadings_level1'])
    assert loaded.assets == universe.assets
    pd.testing.assert_frame_equal(loaded.metadata, universe.metadata)
    pd.testing.assert_frame_equal(loaded.prices, universe.prices,
                                  check_freq=False, atol=1e-8)
    assert loaded.group_loadings_level1 is not None


def test_load_derives_the_metadata_contract_when_none_is_given(tmp_path) -> None:
    """with no enum supplied the loader builds one from the columns it actually read"""
    local_path = f"{tmp_path}/"
    make_universe().save(file_name='derived', local_path=local_path)
    loaded = UniverseData.load(file_name='derived', local_path=local_path)
    assert {field.value for field in loaded.metadata_fields} == set(loaded.metadata.columns)


def test_load_can_restrict_to_a_time_period(tmp_path) -> None:
    """a time period narrows the prices at load time rather than after"""
    universe = make_universe()
    local_path = f"{tmp_path}/"
    universe.save(file_name='windowed', local_path=local_path)
    period = qis.TimePeriod(universe.prices.index[100], universe.prices.index[200])
    loaded = UniverseData.load(file_name='windowed', local_path=local_path,
                               metadata_fields=MetadataField, time_period=period)
    assert len(loaded.prices) < len(universe.prices)
    assert loaded.prices.index[0] >= period.start


def test_save_and_load_round_trip_both_loadings_levels(tmp_path) -> None:
    """two loadings keys are written and read back in the order they are named"""
    level2 = make_group_loadings().rename(columns={'Growth': 'Liquid', 'Defensive': 'Illiquid'})
    universe = make_universe(group_loadings_level2=level2)
    local_path = f"{tmp_path}/"
    universe.save(file_name='two_levels', local_path=local_path)
    loaded = UniverseData.load(
        file_name='two_levels', local_path=local_path, metadata_fields=MetadataField,
        group_loadings_keys=['group_loadings_level1', 'group_loadings_level2'])
    assert list(loaded.group_loadings_level1.columns) == ['Growth', 'Defensive']
    assert list(loaded.group_loadings_level2.columns) == ['Liquid', 'Illiquid']


def test_load_rejects_a_dataset_that_carries_no_metadata(tmp_path) -> None:
    """a missing metadata file yields no contract at all, so the loader stops here

    ``qis.load_df_dict_from_csv`` skips a key whose file is absent rather than raising, so
    without this check the universe would be constructed against a ``None`` metadata table and
    fail much later inside validation with no clue which file was missing.
    """
    local_path = f"{tmp_path}/"
    make_universe().save(file_name='headless', local_path=local_path)
    (tmp_path / 'headless_metadata.csv').unlink()
    with pytest.raises(ValueError, match="No 'metadata' key found"):
        UniverseData.load(file_name='headless', local_path=local_path,
                          metadata_fields=MetadataField)


def test_load_prefers_an_excel_metadata_override_over_the_saved_csv(tmp_path) -> None:
    """metadata_filename replaces the stored metadata, which is how a curated sheet is used"""
    local_path = f"{tmp_path}/"
    make_universe().save(file_name='override', local_path=local_path)
    override = make_metadata()
    override[MetadataField.NAME.value] = ['A', 'B', 'C', 'D']
    override.to_excel(f"{local_path}curated.xlsx", sheet_name='Sheet1')
    loaded = UniverseData.load(file_name='override', local_path=local_path,
                               metadata_fields=MetadataField, metadata_filename='curated')
    assert list(loaded.name) == ['A', 'B', 'C', 'D']


# --------------------------------------------------------------------------- #
# transforms
# --------------------------------------------------------------------------- #
def test_unsmoothing_raises_the_vol_of_the_flagged_asset_only() -> None:
    """AR(1) unsmoothing corrects the smoothed leg and leaves the others untouched"""
    universe = make_universe()
    flags = pd.Series([False, False, False, True], index=ASSETS)
    unsmoothed = copy_universe_data_with_unsmoothed_prices(
        universe_data=universe, assets_for_unsmoothing=flags, freq='QE', unsmooth_span=8,
        warmup_period=2)
    assert unsmoothed.assets == universe.assets
    pd.testing.assert_frame_equal(unsmoothed.metadata, universe.metadata)
    # the untouched assets come back identical
    for asset in ['equity_us', 'equity_eu', 'bond_govt']:
        pd.testing.assert_series_equal(unsmoothed.prices[asset].dropna(),
                                       universe.prices[asset].dropna(), check_freq=False)
    # the flagged one does not
    assert not unsmoothed.prices['private_equity'].equals(universe.prices['private_equity'])


def test_unsmoothing_rejects_a_flag_series_on_a_different_universe() -> None:
    """a misaligned flag series would silently unsmooth the wrong asset"""
    with pytest.raises(ValueError, match='assets_for_unsmoothing index does not match'):
        copy_universe_data_with_unsmoothed_prices(
            universe_data=make_universe(),
            assets_for_unsmoothing=pd.Series([True], index=['not_an_asset']))


def test_unsmoothing_rejects_a_freq_series_on_a_different_universe() -> None:
    """a per-asset frequency is checked on the same footing as the flags"""
    with pytest.raises(ValueError, match='freq Series index does not match'):
        copy_universe_data_with_unsmoothed_prices(
            universe_data=make_universe(),
            assets_for_unsmoothing=pd.Series([False, False, False, True], index=ASSETS),
            freq=pd.Series(['QE'], index=['not_an_asset']))


def test_unsmoothing_nothing_returns_the_source_universe_untouched() -> None:
    """with no asset flagged there is nothing to correct, so the input comes straight back"""
    universe = make_universe()
    unchanged = copy_universe_data_with_unsmoothed_prices(
        universe_data=universe,
        assets_for_unsmoothing=pd.Series(False, index=ASSETS))
    assert unchanged is universe


def test_unsmoothing_accepts_a_per_asset_frequency_series() -> None:
    """a Series freq is narrowed to the flagged assets before it reaches the estimator

    The freq Series is stated over the whole universe, but only the flagged leg is unsmoothed;
    passing the full Series through would ask the estimator for a frequency per asset it was
    never given prices for.
    """
    universe = make_universe()
    flags = pd.Series([False, False, False, True], index=ASSETS)
    freqs = pd.Series(['ME', 'ME', 'ME', 'QE'], index=ASSETS)
    unsmoothed = copy_universe_data_with_unsmoothed_prices(
        universe_data=universe, assets_for_unsmoothing=flags, freq=freqs, unsmooth_span=8,
        warmup_period=2)
    assert unsmoothed.assets == universe.assets
    assert not unsmoothed.prices['private_equity'].equals(universe.prices['private_equity'])
