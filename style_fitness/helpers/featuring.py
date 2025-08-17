# Import modules
import pandas as pd
import prince

from sklearn.feature_selection import mutual_info_regression
from sklearn.impute import SimpleImputer
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Plot modules
import matplotlib.pyplot as plt
import seaborn as sns

# Functions

def calculate_mi(df: pd.DataFrame, y: pd.Series):
    """
    Calculate Mutual Information (MI) for all features in a DataFrame.

    Args:
            df (pd.DataFrame): Input DataFrame.
            y (pd.Series): Target variable.

    Returns:
            pd.DataFrame: DataFrame containing feature names and their corresponding MI scores.
    """
    # Calculate MI for all features
    mi_scores = mutual_info_regression(df, y, random_state=42)
    mi_df = pd.DataFrame({"Features": df.columns, "MI Score": mi_scores})

    # Sort features by MI score in descending order
    mi_df = mi_df.sort_values(by="MI Score", ascending=False).reset_index(
        drop=True
    )

    # Rank the values from 1
    mi_df["Imp Rank"] = mi_df["MI Score"].rank(ascending=False)

    return mi_df


def calculate_vif(df: pd.DataFrame, features: list[str]):
    """
    Calculate Variance Inflation Factor (VIF) for a given DataFrame and list of features.

    Args:
            df (pd.DataFrame): Input DataFrame.
            features (list[str]): List of feature names.

    Returns:
            pd.DataFrame: DataFrame containing feature names and their corresponding VIF scores and ranks.
    """
    # Select only the features and convert to float
    X = df[features].copy()
    # Try to convert all columns to numeric, coerce errors to NaN
    X = X.apply(pd.to_numeric, errors='coerce').astype('float')

    # Check for any remaining non-numeric columns
    non_numeric = X.columns[X.dtypes == 'object']
    if len(non_numeric) > 0:
        raise ValueError(f"Non-numeric columns found: {list(non_numeric)}")

    vif = pd.DataFrame()
    vif["Features"] = features
    vif["VIF Score"] = [
        variance_inflation_factor(X.values, i)
        for i in range(len(features))
    ]
    vif["Elimination rank"] = vif["VIF Score"].rank(ascending=False)

    return vif


def plot_mi_and_vif(df: pd.DataFrame, y: pd.Series, min_features: int = 5):
    """
    Plot mutual information and variance inflation factor (VIF) scores for a given DataFrame and target variable.

    Args:
            df (pd.DataFrame): Input DataFrame.
            y (pd.Series): Target variable.
            min_features (int): Minimum number of features to consider.

    Returns:
            None
    """
    # Initialize results and remaining features
    results = []
    selected_features_history = []
    remaining_features = df.columns.to_list()

    while len(remaining_features) > min_features:
        # --- Handle missing values ---
        imputer = SimpleImputer(strategy="mean")
        current_X = df[remaining_features]
        current_X = pd.DataFrame(
            imputer.fit_transform(current_X), columns=current_X.columns
        )

        mi_df = calculate_mi(current_X, y)
        vif_df = calculate_vif(current_X, remaining_features)

        merged = pd.merge(mi_df, vif_df, on="Features")
        merged["Num Features"] = len(remaining_features)
        results.append(
            merged[["Features", "MI Score", "VIF Score", "Num Features"]]
        )

        # Eliminate the feature with the lowest MI
        lowest_mi_feature = mi_df.iloc[-1]["Features"]
        remaining_features.remove(lowest_mi_feature)
        selected_features_history.append(remaining_features.copy())

    # Combine all iterations
    results_df = pd.concat(results)

    # Aggregate MI and VIF by number of features remaining
    agg_results = (
        results_df.groupby("Num Features")
        .agg({"MI Score": "mean", "VIF Score": ["mean", "min", "max"]})
        .reset_index()
    )
    agg_results.columns = [
        "Num Features",
        "MI Score",
        "VIF Mean",
        "VIF Min",
        "VIF Max",
    ]

    # --- Seaborn plot ---
    sns.set(style="whitegrid")
    fig, ax1 = plt.subplots(figsize=(8, 6))

    # Primary Y-axis: MI
    sns.lineplot(
        data=agg_results,
        x="Num Features",
        y="MI Score",
        marker="o",
        ax=ax1,
        color="royalblue",
        label="Mean MI",
    )

    ax1.set_title(
        "Feature Elimination Impact on MI and VIF Scores", fontsize=12
    )
    ax1.set_xlabel("Number of Features", fontsize=10)
    ax1.set_ylabel("Mean MI Score", fontsize=10, color="royalblue")
    ax1.tick_params(axis="y", labelcolor="royalblue")
    ax1.invert_xaxis()
    ax1.grid(True)

    # Secondary Y-axis: VIF
    ax2 = ax1.twinx()
    sns.lineplot(
        data=agg_results,
        x="Num Features",
        y="VIF Mean",
        ax=ax2,
        color="firebrick",
        label="VIF Mean",
        marker="s",
    )
    sns.lineplot(
        data=agg_results,
        x="Num Features",
        y="VIF Max",
        ax=ax2,
        color="darkred",
        label="VIF Max",
        linestyle="--",
    )
    sns.lineplot(
        data=agg_results,
        x="Num Features",
        y="VIF Min",
        ax=ax2,
        color="salmon",
        label="VIF Min",
        linestyle="--",
    )

    ax2.set_ylabel("VIF Scores", fontsize=10, color="firebrick")
    ax2.tick_params(axis="y", labelcolor="firebrick")

    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=9)
    # Remove legend from right axis
    ax2.legend_.remove() if ax2.legend_ else None

    plt.tight_layout()
    plt.show()

    return selected_features_history


# Helper function to apply FAMD


def apply_famd(df: pd.DataFrame, n_components: int = 1):
    """
    Calculate the FAMD model in `n_components` and return
    the new components/dimentions.

    Args:
        df (pd.DataFrame): Input data
        n_components (int): Number of components to calculate

    Returns:
        famd (prince.famd.FAMD): FAMD model
        df_famd (pd.DataFrame): FAMD-transformed data
    """
    # define FAMD model
    famd = prince.FAMD(
        n_components=n_components,
        n_iter=20,
        copy=True,
        check_input=True,
        random_state=42,
    )

    # Fit FAMD model
    famd = famd.fit(df)

    # Obtain the new components/dimentions
    df_famd = famd.transform(df)

    # Preview results
    print("\nShape of FAMD-transformed data:", df_famd.shape)
    print("First 5 rows of the FAMD-transformed data:")
    print(
        df_famd.head().to_markdown(
            index=False, numalign="left", stralign="left"
        )
    )

    return famd, df_famd


def calc_explained_inertia(famd: prince.famd.FAMD):
    """
    Calculate the explained inertia for each component in the FAMD model.

    Args:
        famd (prince.famd.FAMD): FAMD model

    Returns:
        explained_inertia (np.ndarray): Explained inertia for each component
    """
    # 'famd' is already fitted
    eigvals = famd.eigenvalues_
    explained_inertia = eigvals / eigvals.sum()

    # Print `inertia` for each component
    for i, fraction in enumerate(explained_inertia, start=0):
        print(f"Component {i}: {fraction * 100:.2f}% of inertia explained")

    return explained_inertia


def plot_famd_variance(famd: prince.famd.FAMD, figsize: tuple = (10, 6)):
    """
    Plot the explained and cumulative inertia for each component in the FAMD model.

    Args:
        famd (prince.famd.FAMD): FAMD model
        figsize (tuple): Figure size for the plot

    Returns:
        None
    """
    # 'famd' is already fitted
    explained_inertia = calc_explained_inertia(famd)
    cumulative_inertia = explained_inertia.cumsum()

    # Build a DataFrame for plotting
    inertia_df = pd.DataFrame(
        {
            "Component": [
                f"Dim {i + 1}" for i in range(len(explained_inertia))
            ],
            "Explained Inertia": explained_inertia * 100,
            "Cumulative Inertia": cumulative_inertia * 100,
        }
    )
    # Sort components in descending order
    inertia_df = inertia_df.sort_values(by="Explained Inertia", ascending=False)

    # Plot both in one figure
    fig, ax1 = plt.subplots(figsize=figsize)
    sns.set(style="whitegrid")

    # Barplot (Explained Inertia per component)
    sns.barplot(
        data=inertia_df,
        x="Component",
        y="Explained Inertia",
        ax=ax1,
        color="skyblue",
    )

    # Lineplot (Cumulative Inertia)
    ax2 = ax1.twinx()
    sns.lineplot(
        data=inertia_df,
        x="Component",
        y="Cumulative Inertia",
        ax=ax2,
        marker="o",
        color="firebrick",
        label="Cumulative Inertia",
    )

    # Labeling
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=90)
    ax1.set_ylabel("Explained Inertia (%)", color="steelblue")
    ax2.set_ylabel("Cumulative Inertia (%)", color="firebrick")
    ax1.set_title("FAMD Component Inertia Explained and Cumulative")
    ax2.set_ylim(0, 105)

    # Add cumulative line to legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines2, labels2, loc="upper right")

    plt.tight_layout()
    plt.show()
