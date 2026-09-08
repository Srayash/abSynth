import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass, field
from sklearn.mixture import GaussianMixture
import warnings

warnings.filterwarnings("ignore")


@dataclass
class TVAEPreprocessorConfig:
    """Configuration for TVAE Preprocessor."""

    # Gaussian Mixture Model settings for continuous columns
    n_gmm_components: int = 10
    gmm_covariance_type: str = "diag"
    gmm_max_iter: int = 100
    gmm_random_state: int = 42
    gmm_reg_covar: float = 1e-3

    # Categorical encoding settings
    categorical_encoding: str = "one_hot"  # 'one_hot' or 'label'
    handle_unknown: str = "ignore"  # 'ignore' or 'error'

    # Normalization settings
    normalize_continuous: bool = True
    eps: float = 1e-6  # Small epsilon for numerical stability

    # Column detection settings
    categorical_threshold: int = 20  # Max unique values to consider categorical

    def __post_init__(self):
        if self.categorical_encoding not in ["one_hot", "label"]:
            raise ValueError("categorical_encoding must be 'one_hot' or 'label'")
        if self.handle_unknown not in ["ignore", "error"]:
            raise ValueError("handle_unknown must be 'ignore' or 'error'")


@dataclass
class ContinuousColumnInfo:
    """Stores information about a continuous column."""

    name: str
    gmm: GaussianMixture
    min_value: float
    max_value: float
    output_dim: int  # n_components (for mode) + 1 (for normalized value)


@dataclass
class CategoricalColumnInfo:
    """Stores information about a categorical column."""

    name: str
    categories: List[str]
    category_to_idx: Dict[str, int]
    output_dim: int  # Number of one-hot dimensions


class TVAEPreprocessor:
    """Preprocessor for Tabular VAE that handles mixed-type data.

    For continuous columns:
    - Fits a Gaussian Mixture Model (GMM)
    - Transforms to [mode_indicator (one-hot), normalized_value]

    For categorical columns:
    - One-hot encodes the categories
    """

    def __init__(self, config: Optional[TVAEPreprocessorConfig] = None):
        self.config = config or TVAEPreprocessorConfig()
        self.continuous_columns_info: List[ContinuousColumnInfo] = []
        self.categorical_columns_info: List[CategoricalColumnInfo] = []
        self.column_order: List[str] = []
        self.fitted = False

    def _detect_column_types(
        self,
        df: pd.DataFrame,
        continuous_columns: Optional[List[str]] = None,
        categorical_columns: Optional[List[str]] = None,
    ) -> Tuple[List[str], List[str]]:
        """Detect or validate column types."""

        if continuous_columns is None and categorical_columns is None:
            # Auto-detect based on dtype and unique values
            continuous = []
            categorical = []

            for col in df.columns:
                if pd.api.types.is_numeric_dtype(df[col]):
                    n_unique = df[col].nunique()
                    if n_unique <= self.config.categorical_threshold:
                        categorical.append(col)
                    else:
                        continuous.append(col)
                else:
                    categorical.append(col)

            return continuous, categorical

        elif continuous_columns is not None and categorical_columns is not None:
            # Validate provided columns
            all_cols = set(continuous_columns + categorical_columns)
            if all_cols != set(df.columns):
                raise ValueError("Provided columns don't match dataframe columns")
            return continuous_columns, categorical_columns

        else:
            raise ValueError("Either provide both column lists or neither")

    def fit(
        self,
        df: pd.DataFrame,
        continuous_columns: Optional[List[str]] = None,
        categorical_columns: Optional[List[str]] = None,
    ) -> "TVAEPreprocessor":
        """Fit the preprocessor on the data.

        Args:
            df: Input dataframe
            continuous_columns: List of continuous column names (optional)
            categorical_columns: List of categorical column names (optional)
        """

        continuous_cols, categorical_cols = self._detect_column_types(
            df, continuous_columns, categorical_columns
        )

        self.column_order = list(df.columns)

        # Fit continuous columns with GMM
        for col in continuous_cols:
            data = df[col].values.reshape(-1, 1)

            # Fit Gaussian Mixture Model
            gmm = GaussianMixture(
                n_components=self.config.n_gmm_components,
                covariance_type=self.config.gmm_covariance_type,
                max_iter=self.config.gmm_max_iter,
                random_state=self.config.gmm_random_state,
                reg_covar=self.config.gmm_reg_covar,
            )
            gmm.fit(data)

            info = ContinuousColumnInfo(
                name=col,
                gmm=gmm,
                min_value=float(data.min()),
                max_value=float(data.max()),
                output_dim=self.config.n_gmm_components + 1,
            )
            self.continuous_columns_info.append(info)

        # Fit categorical columns
        for col in categorical_cols:
            categories = sorted(df[col].unique().astype(str).tolist())
            category_to_idx = {cat: idx for idx, cat in enumerate(categories)}

            info = CategoricalColumnInfo(
                name=col,
                categories=categories,
                category_to_idx=category_to_idx,
                output_dim=len(categories),
            )
            self.categorical_columns_info.append(info)

        self.fitted = True
        return self

    def _transform_continuous(
        self, data: np.ndarray, info: ContinuousColumnInfo
    ) -> np.ndarray:
        """Transform a continuous column using its GMM.

        Returns: [mode_one_hot (n_components), normalized_value (1)]
        """
        data = data.reshape(-1, 1)
        n_samples = len(data)

        # Get GMM component assignments
        modes = info.gmm.predict(data)

        # One-hot encode the modes
        mode_one_hot = np.zeros((n_samples, self.config.n_gmm_components))
        mode_one_hot[np.arange(n_samples), modes] = 1.0

        # Normalize values to [0, 1] within each component
        normalized_values = np.zeros((n_samples, 1))

        for mode_idx in range(self.config.n_gmm_components):
            mask = modes == mode_idx
            if mask.sum() == 0:
                continue

            # Get mean and std of this component
            mean = info.gmm.means_[mode_idx, 0]
            std = np.sqrt(info.gmm.covariances_[mode_idx, 0])

            # Normalize: (x - mean) / (4 * std), then clip to [-0.99, 0.99]
            normalized = (data[mask, 0] - mean) / (4 * std + self.config.eps)
            normalized = np.clip(normalized, -0.99, 0.99)
            normalized_values[mask, 0] = normalized

        return np.concatenate([mode_one_hot, normalized_values], axis=1)

    def _transform_categorical(
        self, data: np.ndarray, info: CategoricalColumnInfo
    ) -> np.ndarray:
        """One-hot encode a categorical column."""
        n_samples = len(data)
        one_hot = np.zeros((n_samples, info.output_dim))

        for i, value in enumerate(data):
            value_str = str(value)
            if value_str in info.category_to_idx:
                idx = info.category_to_idx[value_str]
                one_hot[i, idx] = 1.0
            elif self.config.handle_unknown == "error":
                raise ValueError(
                    f"Unknown category '{value_str}' in column {info.name}"
                )
            # else: handle_unknown == 'ignore', leave as all zeros

        return one_hot

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transform the data.

        Returns: Transformed array of shape (n_samples, total_output_dim)
        """
        if not self.fitted:
            raise RuntimeError("Preprocessor must be fitted before transform")

        # Reorder columns to match training order
        df = df[self.column_order]

        transformed_parts = []

        # Transform continuous columns
        for info in self.continuous_columns_info:
            data = df[info.name].values
            transformed = self._transform_continuous(data, info)
            transformed_parts.append(transformed)

        # Transform categorical columns
        for info in self.categorical_columns_info:
            data = df[info.name].values
            transformed = self._transform_categorical(data, info)
            transformed_parts.append(transformed)

        return np.concatenate(transformed_parts, axis=1)

    def fit_transform(
        self,
        df: pd.DataFrame,
        continuous_columns: Optional[List[str]] = None,
        categorical_columns: Optional[List[str]] = None,
    ) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(df, continuous_columns, categorical_columns)
        return self.transform(df)

    def inverse_transform(self, transformed_data: np.ndarray) -> pd.DataFrame:
        """Inverse transform the data back to original space.

        Args:
            transformed_data: Transformed array from transform()

        Returns: DataFrame with original column names and types
        """
        if not self.fitted:
            raise RuntimeError("Preprocessor must be fitted before inverse_transform")

        n_samples = len(transformed_data)
        result = {}
        current_idx = 0

        # Inverse transform continuous columns
        for info in self.continuous_columns_info:
            # Extract mode one-hot and normalized value
            mode_one_hot = transformed_data[
                :, current_idx : current_idx + self.config.n_gmm_components
            ]
            normalized_values = transformed_data[
                :, current_idx + self.config.n_gmm_components
            ]
            current_idx += info.output_dim

            # Get the selected mode for each sample
            modes = np.argmax(mode_one_hot, axis=1)

            # Reconstruct values
            reconstructed = np.zeros(n_samples)
            for mode_idx in range(self.config.n_gmm_components):
                mask = modes == mode_idx
                if mask.sum() == 0:
                    continue

                mean = info.gmm.means_[mode_idx, 0]
                std = np.sqrt(info.gmm.covariances_[mode_idx, 0])

                # Inverse of normalization
                values = normalized_values[mask] * 4 * std + mean
                reconstructed[mask] = values

            # Clip to original range
            reconstructed = np.clip(reconstructed, info.min_value, info.max_value)
            result[info.name] = reconstructed

        # Inverse transform categorical columns
        for info in self.categorical_columns_info:
            one_hot = transformed_data[:, current_idx : current_idx + info.output_dim]
            current_idx += info.output_dim

            # Get the category with highest probability
            category_indices = np.argmax(one_hot, axis=1)
            categories = [info.categories[idx] for idx in category_indices]
            result[info.name] = categories

        return pd.DataFrame(result)[self.column_order]

    def get_output_dim(self) -> int:
        """Get the total output dimension after transformation."""
        if not self.fitted:
            raise RuntimeError("Preprocessor must be fitted first")

        total = sum(info.output_dim for info in self.continuous_columns_info)
        total += sum(info.output_dim for info in self.categorical_columns_info)
        return total

    def get_output_info(self) -> List[Tuple[int, int, str]]:
        """Describe the layout of the transformed output for loss computation.

        Returns a list of (start_idx, dim, kind) segments, where kind is
        'discrete' for one-hot blocks (GMM mode indicators + categorical
        one-hot columns) that should use cross-entropy loss, and
        'continuous' for single normalized-value dims that should use MSE.
        """
        if not self.fitted:
            raise RuntimeError("Preprocessor must be fitted first")

        output_info = []
        idx = 0

        for info in self.continuous_columns_info:
            n_components = self.config.n_gmm_components
            output_info.append((idx, n_components, "discrete"))  # mode one-hot
            output_info.append(
                (idx + n_components, 1, "continuous")
            )  # normalized value
            idx += info.output_dim

        for info in self.categorical_columns_info:
            output_info.append((idx, info.output_dim, "discrete"))  # category one-hot
            idx += info.output_dim

        return output_info

    def get_metadata(self) -> Dict:
        """Get metadata about the fitted preprocessor."""
        if not self.fitted:
            raise RuntimeError("Preprocessor must be fitted first")

        return {
            "continuous_columns": [info.name for info in self.continuous_columns_info],
            "categorical_columns": [
                info.name for info in self.categorical_columns_info
            ],
            "output_dim": self.get_output_dim(),
            "continuous_dims": {
                info.name: info.output_dim for info in self.continuous_columns_info
            },
            "categorical_dims": {
                info.name: info.output_dim for info in self.categorical_columns_info
            },
        }
