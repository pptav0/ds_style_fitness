# data modules
import pandas as pd

# plot modules
import seaborn as sns
import matplotlib.pyplot as plt

def convert_binary(value, yes_values:set={'yes', '1', 1, True}, no_values:set = {'no', '0', 0, False}):
    """
    Converts various representations of set status to binary.
    Returns 1 for 'yes'/'1', 0 for 'no'/'0', and None for unknown values.
    """
    # Normalize to string and lowercase for comparison
    val_str = str(value).strip().lower()

    if val_str in yes_values or value in yes_values:
        return True
    elif val_str in no_values or value in no_values:
        return False
    else:
        raise ValueError("Value not recognized")


def plot_boxplot(data:pd.DataFrame, figsize:tuple=(12,4)):
	'''
	This function will create a boxplot for each data column.
	`data` is taken as it is, no modification or filter is processed
	inside the function.
	'''
	# get data Columns
	cols = data.columns.tolist()

	# create the subplots
	fig, axes = plt.subplots(nrows=1, ncols=len(cols), figsize=figsize)
	fig.tight_layout(pad=5)

	# iterate through each column
	for i,col in enumerate(cols):
		sns.boxplot(data=data[col], ax=axes[i], fill=False, gap=0.1)

	# print descriptive statistics
	print(data.describe())
	print()

	# show the plot
	plt.show()

# Let's see the distribution of the data using some helper functions

def calc_iqr_bounds(df: pd.Series, threshold:float=1.5):
  """
  Calculate the Interquartile Range (IQR) Lower and Upper bounds.
  """
  # calc inter-quartile range
  q1 = df.quantile(0.25)
  q3 = df.quantile(0.75)
  iqr = q3 - q1

  # outlier boundaries
  lower_bound = q1 - (threshold * iqr)
  upper_bound = q3 + (threshold * iqr)

  return lower_bound, upper_bound


def plot_boxplot_and_outliers(df: pd.DataFrame, col: str, threshold:float=0.05, figsize: tuple=(12, 4)):
  """
  Plot histogram, boxplot, and scatter plot with frequency on the x-axis.
  """
  # Create the plot layout
  fig, axes = plt.subplots(1, 3, figsize=figsize, sharey=True)

  # Histogram
  col_mean = df[col].mean()
  sns.histplot(y=df[col], kde=True,
              color='blue', ax=axes[0])

  axes[0].axhline(col_mean, color='r', linestyle='--',
                  label=f'median = {col_mean: .2f}')

  axes[0].set_title(f'Histogram ({col})')
  axes[0].set_xlabel('Frequency')
  axes[0].set_ylabel(f'{col}')
  axes[0].legend()

  # Boxplot of data
  sns.boxplot(y=df[col], color='green', width=0.5, ax=axes[1])
  axes[1].set_title(f'Boxplot ({col})')
  axes[1].set_xlabel('Data')

  # Scatter plot of data frequency
  plt.subplot(1, 3, 3)
  sns.scatterplot(x=df.index, y=df[col],
                  color='g', label=f'{col}', ax=axes[2])
  # Outliers:
  lower_bound, upper_bound = calc_iqr_bounds(df[col])
  outliers = df[col][(df[col] < lower_bound) | (df[col] > upper_bound)]
  if not outliers.empty:  # Ensure outliers exist before plotting
    sns.scatterplot(
        x=outliers.index, y=outliers.values,
        color='r', label='Outliers', ax=axes[2])


  # Quantile thresholds
  q_lower = df[col].quantile(threshold)
  q_upper = df[col].quantile(1-threshold)

  # Draw horizontal lines for thresholds
  axes[2].axhline(q_lower, color='r', linestyle='--',
                  label=f'{int(threshold*100)}th percentile')
  axes[2].axhline(q_upper, color='r', linestyle='--',
                  label=f'{int((1-threshold)*100)}th percentile') # Red line at Threshold
  axes[2].set_title(f'Scatterplot of ({col})')
  axes[2].set_xlabel('Index')
  axes[2].set_ylabel('Frequency')
  axes[2].legend()

  plt.tight_layout()
  plt.show()


def plot_correlation(data: pd.DataFrame, figsize:tuple=(8, 6)):
	# Compute correlation matrix (numeric columns only)
	corr_matrix = data.corr(numeric_only=True)

	# Set up the matplotlib figure
	plt.figure(figsize=figsize)

	# Draw the heatmap
	sns.heatmap(
	    corr_matrix,
	    annot=True,        # Show correlation values
	    fmt=".2f",         # Format values
	    cmap="coolwarm",   # Color map
	    cbar=True,         # Show color bar
	    linewidths=0.5
	)

	plt.title("Correlation Heatmap", fontsize=14)
	plt.show()
