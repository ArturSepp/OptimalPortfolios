"""
Investment universe universe structures for portfolio optimisation.

This module defines the cmas universe container `UniverseData` that pairs
asset price histories with descriptive metadata. This pattern of
"paired DataFrames" (prices as columns, metadata as index) is used
consistently across the optimalportfolios package and its companion
library qis (QuantInvestStrats).

Typical workflow:
    1. Load or construct a UniverseData instance for your investment universe
    2. Pass it to covariance estimation (covar_estimation module)
    3. Feed estimates into portfolio optimisation (optimization module)
    4. Generate reports using optimised weights and universe metadata (reports module)

Example:
    >>> from optimalportfolios.universe import UniverseData, MetadataField
    >>>
    >>> # Load from CSV
    >>> universe = UniverseData.load(file_name='global_saa', local_path='./universe')
    >>>
    >>> # Or construct directly
    >>> universe = UniverseData(prices=prices_df, metadata=metadata_df)
    >>>
    >>> # Subset to specific assets
    >>> em_universe = UniverseData.from_selection(
    ...     prices=prices_df,
    ...     metadata=metadata_df,
    ...     assets=['EM Equity', 'EM Bonds', 'EM FX'],
    ... )

The metadata DataFrame must contain columns defined by a MetadataField enum.
The default enum requires 'asset_class' and 'currency' columns, but callers
can pass a custom enum to enforce domain-specific schemas (e.g., SAA vs TAA
universes with different required attributes).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Type, List
import pandas as pd
import qis as qis


class MetadataField(str, Enum):
    """Base metadata fields. Subclass or replace for specific universes."""
    NAME = 'name'
    ASSET_CLASS = 'asset_class'
    CURRENCY = 'currency'


@dataclass(frozen=True)
class UniverseData:
    """Immutable container for a single investment universe.

    Attributes:
        prices: Asset prices. DatetimeIndex, columns = asset names.
        metadata: Asset attributes. Index = asset names.
        metadata_fields: Enum class defining required metadata columns.
        group_loadings_level1: Asset-to-group binary mapping for level 1.
        group_loadings_level2: Asset-to-group binary mapping for level 2.
    """
    prices: pd.DataFrame
    metadata: pd.DataFrame
    metadata_fields: Type[Enum] = MetadataField
    group_loadings_level1: Optional[pd.DataFrame] = field(default=None)
    group_loadings_level2: Optional[pd.DataFrame] = field(default=None)

    # ac identifiers
    liquidity_ac_id: str = 'Liquidity'
    equity_ac_id: str = 'Equities'
    bond_ac_id: str = 'Bonds'
    pe_asset_id: Optional[str] = None

    # validation control
    validate_on_init: bool = field(default=True, repr=False)

    def __post_init__(self):
        if self.validate_on_init:
            self.validate()

    def rename_index(self) -> UniverseData:
        rename_map = self.name.to_dict()
        prices = self.prices.rename(rename_map, axis=1)
        metadata = self.metadata.rename(rename_map, axis=0)
        if self.group_loadings_level1 is not None:
            group_loadings_level1 = self.group_loadings_level1.rename(rename_map, axis=0)
        else:
            group_loadings_level1 = None
        if self.group_loadings_level2 is not None:
            group_loadings_level2 = self.group_loadings_level2.rename(rename_map, axis=0)
        else:
            group_loadings_level2 = None

        return UniverseData(prices=prices, metadata=metadata, metadata_fields=self.metadata_fields,
                            group_loadings_level1=group_loadings_level1, group_loadings_level2=group_loadings_level2)


    def validate(self) -> None:
        """Run all consistency checks. Raises ValueError on failure."""
        self._validate_asset_alignment()
        self._validate_metadata_fields()
        self._validate_no_duplicates()
        self._validate_no_nulls_in_metadata()
        if self.group_loadings_level1 is not None or self.group_loadings_level2 is not None:
            self._validate_group_loadings()

    def _validate_asset_alignment(self) -> None:
        price_assets = set(self.prices.columns)
        meta_assets = set(self.metadata.index)
        if price_assets != meta_assets:
            missing_in_meta = price_assets - meta_assets
            missing_in_prices = meta_assets - price_assets
            raise ValueError(
                f"Asset mismatch: in prices not metadata={missing_in_meta}, "
                f"in metadata not prices={missing_in_prices}"
            )

    def _validate_metadata_fields(self) -> None:
        required = {f.value for f in self.metadata_fields}
        present = set(self.metadata.columns)
        missing = required - present
        if missing:
            raise ValueError(f"Metadata missing required columns: {missing}")

    def _validate_group_loadings(self) -> None:
        price_assets = set(self.prices.columns)
        for name, gl in [('level1', self.group_loadings_level1),
                         ('level2', self.group_loadings_level2)]:
            if gl is not None and set(gl.index) != price_assets:
                raise ValueError(
                    f"group_loadings_{name} index doesn't match price columns"
                )

    def _validate_no_duplicates(self) -> None:
        if self.prices.columns.duplicated().any():
            dupes = self.prices.columns[self.prices.columns.duplicated()].tolist()
            raise ValueError(f"Duplicate asset names in prices: {dupes}")
        if self.metadata.index.duplicated().any():
            dupes = self.metadata.index[self.metadata.index.duplicated()].tolist()
            raise ValueError(f"Duplicate asset names in metadata: {dupes}")

    def _validate_no_nulls_in_metadata(self) -> None:
        required = {f.value for f in self.metadata_fields}
        nulls = self.metadata[list(required)].isnull().any()
        cols_with_nulls = nulls[nulls].index.tolist()
        if cols_with_nulls:
            raise ValueError(f"Null values in required metadata columns: {cols_with_nulls}")

    @property
    def name(self) -> pd.Series:
        return self.metadata[MetadataField.NAME]

    @property
    def asset_class(self) -> pd.Series:
        return self.metadata[MetadataField.ASSET_CLASS]

    @property
    def currency(self) -> pd.Series:
        return self.metadata[MetadataField.CURRENCY]

    @property
    def assets(self) -> list[str]:
        return list(self.prices.columns)

    @property
    def n_assets(self) -> int:
        return len(self.prices.columns)

    @property
    def date_range(self) -> tuple[pd.Timestamp, pd.Timestamp]:
        return self.prices.index[0], self.prices.index[-1]

    @classmethod
    def from_selection(
        cls,
        prices: pd.DataFrame,
        metadata: pd.DataFrame,
        assets: list[str],
        metadata_fields: Type[Enum] = MetadataField,
        group_loadings_level1: Optional[pd.DataFrame] = None,
        group_loadings_level2: Optional[pd.DataFrame] = None,
    ) -> UniverseData:
        """Subset a broader dataset to selected assets with validation."""
        return cls(
            prices=prices[assets],
            metadata=metadata.loc[assets],
            metadata_fields=metadata_fields,
            group_loadings_level1=group_loadings_level1.loc[assets] if group_loadings_level1 is not None else None,
            group_loadings_level2=group_loadings_level2.loc[assets] if group_loadings_level2 is not None else None,
        )

    @classmethod
    def load(
            cls,
            file_name: str,
            local_path: str,
            metadata_fields: Optional[Type[Enum]] = None,
            group_loadings_keys: Optional[list[str]] = None,
            metadata_filename: str = None,
            time_period: qis.TimePeriod = None,
            equity_ac_id: str = 'Equities',
            bond_ac_id: str = 'Bonds',
            pe_asset_id: Optional[str] = None,
            validate_on_init: bool = True
    ) -> UniverseData:
        """Load UniverseData from CSV files.
        """
        dataset_keys = ['prices', 'metadata']
        if group_loadings_keys is not None:
            dataset_keys.extend(group_loadings_keys)

        datasets = qis.load_df_dict_from_csv(
            dataset_keys=dataset_keys,
            file_name=file_name,
            local_path=local_path,
        )

        metadata = datasets.get('metadata')
        if metadata is None:
            raise ValueError("No 'metadata' key found in CSV datasets")

        if metadata_fields is None:
            metadata_fields = Enum(
                'MetadataField',
                {col.upper(): col for col in metadata.columns}
            )
        # load with generic fields
        if metadata_filename is not None:
            metadata = qis.load_df_from_excel(file_name=metadata_filename, local_path=local_path)

        gl1 = None
        gl2 = None
        if group_loadings_keys:
            gl1 = datasets.get(group_loadings_keys[0])
            if len(group_loadings_keys) > 1:
                gl2 = datasets.get(group_loadings_keys[1])

        prices = datasets['prices']
        if time_period is not None:
            prices = time_period.locate(prices)

        return cls(
            prices=prices,
            metadata=metadata,
            metadata_fields=metadata_fields,
            group_loadings_level1=gl1,
            group_loadings_level2=gl2,
            equity_ac_id=equity_ac_id,
            bond_ac_id=bond_ac_id,
            pe_asset_id=pe_asset_id,
            validate_on_init=validate_on_init
        )

    def save(self, file_name: str, local_path: str) -> None:
        """Save UniverseData to CSV files."""
        datasets = {
            'prices': self.prices,
            'metadata': self.metadata,
        }
        if self.group_loadings_level1 is not None:
            datasets['group_loadings_level1'] = self.group_loadings_level1
        if self.group_loadings_level2 is not None:
            datasets['group_loadings_level2'] = self.group_loadings_level2

        qis.save_df_dict_to_csv(
            datasets=datasets,
            file_name=file_name,
            local_path=local_path,
        )

    def get_hedge_ratio(self, hedged_acs: List[str]) -> pd.Series:
        """ get hedge ratio for given asset classes"""
        return self.asset_class.isin(hedged_acs).astype(float)
