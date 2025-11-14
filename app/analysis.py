"""
Statistical Data Analysis Module
Client-side Python analysis using PyScript/Pyodide
"""

import pandas as pd
import numpy as np
from io import StringIO, BytesIO
import json
from js import console, document, window, Blob, URL
from pyodide.ffi import create_proxy
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from scipy import stats
from scipy.stats import probplot
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import warnings
warnings.filterwarnings('ignore')

# ========== Global State ==========
class DataState:
    def __init__(self):
        self.df = None
        self.original_df = None
        self.selected_columns = []
        self.dv = None
        self.ivs = []
        self.outlier_info = {}
        self.correlation_results = {}

state = DataState()

# ========== Data Loading Functions ==========
def load_csv_data(file_content):
    """Load CSV data from file content"""
    try:
        console.log("Loading CSV data...")
        df = pd.read_csv(StringIO(file_content))
        console.log(f"Loaded {len(df)} rows and {len(df.columns)} columns")
        return df
    except Exception as e:
        console.error(f"Error loading CSV: {str(e)}")
        window.appFunctions.showError(f"Error loading CSV: {str(e)}")
        return None

def load_excel_data(file_content):
    """Load Excel data from file content"""
    try:
        console.log("Loading Excel data...")
        df = pd.read_excel(BytesIO(file_content))
        console.log(f"Loaded {len(df)} rows and {len(df.columns)} columns")
        return df
    except Exception as e:
        console.error(f"Error loading Excel: {str(e)}")
        window.appFunctions.showError(f"Error loading Excel: {str(e)}")
        return None

def process_uploaded_file():
    """Process the uploaded file and populate variable selections"""
    try:
        console.log("Starting file processing...")
        window.appFunctions.showProgress("Processing uploaded file...", 10)

        # Check if file content exists
        if not hasattr(window, 'fileContent') or not hasattr(window, 'fileName'):
            console.error("File content not found in window")
            window.appFunctions.hideProgress()
            window.appFunctions.showError("File content not found. Please upload the file again.")
            return

        file_content = window.fileContent
        file_name = window.fileName

        console.log(f"Processing file: {file_name}")
        window.appFunctions.updateProgress(30, "Parsing data...")

        if file_name.endswith('.csv'):
            state.df = load_csv_data(file_content)
        else:
            state.df = load_excel_data(file_content)

        if state.df is not None:
            window.appFunctions.updateProgress(60, "Filtering numeric columns...")
            state.original_df = state.df.copy()

            # Filter only numeric columns
            numeric_cols = state.df.select_dtypes(include=[np.number]).columns.tolist()

            console.log(f"Found {len(numeric_cols)} numeric columns: {numeric_cols}")

            if len(numeric_cols) == 0:
                window.appFunctions.hideProgress()
                window.appFunctions.showError("No numeric columns found in the dataset")
                return

            window.appFunctions.updateProgress(90, "Populating variable selections...")

            # Populate variable selections
            window.appFunctions.populateVariableSelections(numeric_cols)

            window.appFunctions.updateProgress(100, "Complete!")
            console.log("Data loaded and variables populated successfully")

            # Hide progress after a short delay
            import js
            js.setTimeout(window.appFunctions.hideProgress, 1000)

        else:
            window.appFunctions.hideProgress()
            window.appFunctions.showError("Failed to load data from file")

    except Exception as e:
        console.error(f"Error processing file: {str(e)}")
        import traceback
        console.error(traceback.format_exc())
        window.appFunctions.hideProgress()
        window.appFunctions.showError(f"Error processing file: {str(e)}")

# ========== Data Quality Check Functions ==========
def check_data_quality():
    """Analyze data quality and display summary statistics"""
    try:
        if state.df is None:
            return

        # Filter to selected variables only
        dv = window.appState.selectedDV
        ivs = window.appState.selectedIVs
        selected_vars = [dv] + list(ivs)

        df_selected = state.df[selected_vars]
        state.selected_columns = selected_vars

        # Dataset summary
        summary_html = f"""
        <div class="alert alert-info">
            <p><strong>Number of Rows:</strong> {len(df_selected)}</p>
            <p><strong>Number of Selected Variables:</strong> {len(selected_vars)}</p>
            <p><strong>Data Types:</strong> All numeric</p>
        </div>
        """
        document.getElementById('data-summary').innerHTML = summary_html

        # Descriptive statistics
        desc_stats = df_selected.describe().round(3)
        desc_html = desc_stats.to_html(classes='table')
        document.getElementById('descriptive-stats').innerHTML = desc_html

        # Check for missing data
        missing_count = df_selected.isnull().sum()
        total_missing = missing_count.sum()

        if total_missing > 0:
            missing_info_html = f"""
            <div class="alert alert-warning">
                <p><strong>Missing Values Detected:</strong> {total_missing} total missing values</p>
            </div>
            <table>
                <thead>
                    <tr><th>Variable</th><th>Missing Count</th><th>Percentage</th></tr>
                </thead>
                <tbody>
            """

            for var in selected_vars:
                missing = missing_count[var]
                if missing > 0:
                    pct = (missing / len(df_selected)) * 100
                    missing_info_html += f"<tr><td>{var}</td><td>{missing}</td><td>{pct:.2f}%</td></tr>"

            missing_info_html += "</tbody></table>"
            document.getElementById('missing-data-info').innerHTML = missing_info_html
            document.getElementById('missing-data-options').classList.remove('hidden')

            # Prepare detailed missing data information
            prepare_missing_data_details(df_selected, selected_vars)
        else:
            missing_info_html = '<div class="alert alert-success">No missing values detected</div>'
            document.getElementById('missing-data-info').innerHTML = missing_info_html
            document.getElementById('missing-data-options').classList.add('hidden')

    except Exception as e:
        console.error(f"Error in data quality check: {str(e)}")
        window.appFunctions.showError(f"Error in data quality check: {str(e)}")

def prepare_missing_data_details(df, variables):
    """Prepare detailed missing data information"""
    try:
        missing_details = []

        for var in variables:
            missing_mask = df[var].isnull()
            missing_indices = missing_mask[missing_mask].index.tolist()

            for idx in missing_indices:
                missing_details.append({
                    'row': idx + 1,  # 1-indexed for user display
                    'column': var,
                    'variable': var,
                    'errorType': 'Missing'
                })

        # Store for modal display
        window.missingDataDetails = missing_details

        # Create imputation options UI
        if missing_details:
            create_imputation_ui(variables)
    except Exception as e:
        console.error(f"Error preparing missing data details: {str(e)}")

def create_imputation_ui(variables):
    """Create UI for imputation options"""
    impute_container = document.getElementById('impute-variables')
    impute_html = ""

    for var in variables:
        impute_html += f"""
        <div class="impute-variable-row">
            <label><strong>{var}:</strong></label>
            <select id="impute-method-{var}" class="form-select">
                <option value="mean">Mean</option>
                <option value="median">Median</option>
                <option value="mode">Mode</option>
            </select>
        </div>
        """

    impute_container.innerHTML = impute_html

def handle_missing_data(method):
    """Handle missing data based on selected method"""
    try:
        df_selected = state.df[state.selected_columns]

        if method == 'delete':
            # Delete rows with any missing values
            df_cleaned = df_selected.dropna()
            console.log(f"Deleted {len(df_selected) - len(df_cleaned)} rows with missing values")

        elif method == 'impute':
            # Impute based on user selection for each variable
            df_cleaned = df_selected.copy()

            for var in state.selected_columns:
                impute_method_elem = document.getElementById(f'impute-method-{var}')
                if impute_method_elem:
                    impute_method = impute_method_elem.value

                    if df_cleaned[var].isnull().sum() > 0:
                        if impute_method == 'mean':
                            df_cleaned[var].fillna(df_cleaned[var].mean(), inplace=True)
                        elif impute_method == 'median':
                            df_cleaned[var].fillna(df_cleaned[var].median(), inplace=True)
                        elif impute_method == 'mode':
                            mode_val = df_cleaned[var].mode()[0] if len(df_cleaned[var].mode()) > 0 else df_cleaned[var].mean()
                            df_cleaned[var].fillna(mode_val, inplace=True)

        elif method == 'manual':
            # Show modal with missing data details
            window.appFunctions.showMissingDataModal(window.missingDataDetails)
            return

        # Update state with cleaned data
        state.df = df_cleaned
        window.appFunctions.showSuccess(f"Data cleaned using {method} method")

    except Exception as e:
        console.error(f"Error handling missing data: {str(e)}")
        window.appFunctions.showError(f"Error handling missing data: {str(e)}")

# ========== Outlier Detection Functions ==========
def detect_outliers_iqr(series):
    """Detect outliers using IQR method"""
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = (series < lower_bound) | (series > upper_bound)
    return outliers, lower_bound, upper_bound

def detect_outliers_zscore(series):
    """Detect outliers using Z-score method"""
    z_scores = np.abs(stats.zscore(series.dropna()))
    outliers_idx = z_scores > 3
    # Create boolean series matching original series length
    outliers = pd.Series(False, index=series.index)
    outliers[series.dropna().index[outliers_idx]] = True
    return outliers

def detect_outliers_modified_z(series):
    """Detect outliers using Modified Z-score (MAD) method"""
    median = series.median()
    mad = np.median(np.abs(series - median))
    modified_z_scores = 0.6745 * (series - median) / mad if mad != 0 else 0
    outliers = np.abs(modified_z_scores) > 3.5
    return outliers

def detect_outliers_in_data(method):
    """Detect outliers in selected variables"""
    try:
        df_selected = state.df[state.selected_columns]
        outlier_summary_html = "<table><thead><tr><th>Variable</th><th>Outliers Count</th><th>Percentage</th></tr></thead><tbody>"

        state.outlier_info = {}

        for var in state.selected_columns:
            if method == 'iqr':
                outliers, lower, upper = detect_outliers_iqr(df_selected[var])
                state.outlier_info[var] = {
                    'outliers': outliers,
                    'lower_bound': lower,
                    'upper_bound': upper,
                    'method': 'iqr'
                }
            elif method == 'zscore':
                outliers = detect_outliers_zscore(df_selected[var])
                state.outlier_info[var] = {
                    'outliers': outliers,
                    'method': 'zscore'
                }
            elif method == 'modified-z':
                outliers = detect_outliers_modified_z(df_selected[var])
                state.outlier_info[var] = {
                    'outliers': outliers,
                    'method': 'modified-z'
                }

            outlier_count = outliers.sum()
            outlier_pct = (outlier_count / len(df_selected)) * 100

            outlier_summary_html += f"<tr><td>{var}</td><td>{outlier_count}</td><td>{outlier_pct:.2f}%</td></tr>"

        outlier_summary_html += "</tbody></table>"
        document.getElementById('outlier-summary').innerHTML = outlier_summary_html

        # Create outlier handling UI
        create_outlier_handling_ui()

    except Exception as e:
        console.error(f"Error detecting outliers: {str(e)}")
        window.appFunctions.showError(f"Error detecting outliers: {str(e)}")

def create_outlier_handling_ui():
    """Create UI for outlier handling options"""
    outlier_container = document.getElementById('outlier-variables')
    outlier_html = ""

    for var in state.selected_columns:
        outlier_count = state.outlier_info[var]['outliers'].sum()

        outlier_html += f"""
        <div class="outlier-variable-card">
            <h4>{var} ({outlier_count} outliers)</h4>
            <div class="outlier-method-select">
                <label>Handling Method:</label>
                <select id="outlier-method-{var}" class="form-select">
                    <option value="ignore">Ignore</option>
                    <option value="delete">Delete Rows</option>
                    <option value="winsorize">Winsorize</option>
                    <option value="replace-mean">Replace with Mean</option>
                    <option value="replace-median">Replace with Median</option>
                    <option value="log">Log Transform</option>
                    <option value="sqrt">Square Root Transform</option>
                </select>
            </div>
        </div>
        """

    outlier_container.innerHTML = outlier_html

def apply_outlier_methods(methods_dict):
    """Apply outlier handling methods"""
    try:
        df_cleaned = state.df.copy()
        rows_to_delete = set()

        for var, method in methods_dict.items():
            if var not in state.outlier_info:
                continue

            outliers = state.outlier_info[var]['outliers']

            if method == 'ignore':
                continue

            elif method == 'delete':
                # Collect indices to delete
                outlier_indices = outliers[outliers].index
                rows_to_delete.update(outlier_indices)

            elif method == 'winsorize':
                if 'lower_bound' in state.outlier_info[var]:
                    lower = state.outlier_info[var]['lower_bound']
                    upper = state.outlier_info[var]['upper_bound']
                    df_cleaned[var] = df_cleaned[var].clip(lower=lower, upper=upper)

            elif method == 'replace-mean':
                non_outlier_mean = df_cleaned.loc[~outliers, var].mean()
                df_cleaned.loc[outliers, var] = non_outlier_mean

            elif method == 'replace-median':
                non_outlier_median = df_cleaned.loc[~outliers, var].median()
                df_cleaned.loc[outliers, var] = non_outlier_median

            elif method == 'log':
                # Add small constant if there are non-positive values
                min_val = df_cleaned[var].min()
                if min_val <= 0:
                    df_cleaned[var] = np.log(df_cleaned[var] - min_val + 1)
                else:
                    df_cleaned[var] = np.log(df_cleaned[var])

            elif method == 'sqrt':
                # Add small constant if there are negative values
                min_val = df_cleaned[var].min()
                if min_val < 0:
                    df_cleaned[var] = np.sqrt(df_cleaned[var] - min_val)
                else:
                    df_cleaned[var] = np.sqrt(df_cleaned[var])

        # Delete collected rows
        if rows_to_delete:
            df_cleaned = df_cleaned.drop(index=list(rows_to_delete))

        state.df = df_cleaned
        window.appFunctions.showSuccess(f"Outlier handling applied. Dataset now has {len(df_cleaned)} rows.")

    except Exception as e:
        console.error(f"Error applying outlier methods: {str(e)}")
        window.appFunctions.showError(f"Error applying outlier methods: {str(e)}")

# ========== Univariate Analysis Functions ==========
def generate_univariate_plots(variable):
    """Generate KDE, Box, and Q-Q plots for a variable"""
    try:
        console.log(f"Generating univariate plots for variable: {variable}")

        if state.df is None:
            console.error("DataFrame is None")
            window.appFunctions.showError("No data loaded")
            return

        if variable not in state.df.columns:
            console.error(f"Variable {variable} not found in dataframe")
            window.appFunctions.showError(f"Variable {variable} not found")
            return

        data = state.df[variable].dropna()
        console.log(f"Data for {variable}: {len(data)} non-null values")

        # KDE Plot
        console.log("Generating KDE plot...")
        fig, ax = plt.subplots(figsize=(8, 5))
        data.plot(kind='kde', ax=ax, color='#2563eb', linewidth=2)
        ax.set_xlabel(variable, fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_title(f'Kernel Density Estimate - {variable}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        display(fig, target='kde-plot')
        plt.close(fig)
        console.log("KDE plot generated successfully")

        # Box Plot
        console.log("Generating box plot...")
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.boxplot(data, vert=True, patch_artist=True,
                   boxprops=dict(facecolor='#3b82f6', alpha=0.7),
                   medianprops=dict(color='#dc2626', linewidth=2),
                   whiskerprops=dict(color='#475569', linewidth=1.5),
                   capprops=dict(color='#475569', linewidth=1.5))
        ax.set_ylabel(variable, fontsize=12)
        ax.set_title(f'Box Plot - {variable}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        display(fig, target='box-plot')
        plt.close(fig)
        console.log("Box plot generated successfully")

        # Q-Q Plot
        console.log("Generating Q-Q plot...")
        distribution = document.getElementById('qq-distribution').value
        generate_qq_plot(variable, distribution)

        window.appFunctions.hideProgress()
        console.log("All univariate plots generated successfully")

    except Exception as e:
        console.error(f"Error generating univariate plots: {str(e)}")
        import traceback
        console.error(traceback.format_exc())
        window.appFunctions.hideProgress()
        window.appFunctions.showError(f"Error generating plots: {str(e)}")

def generate_qq_plot(variable, distribution='norm'):
    """Generate Q-Q plot with specified distribution"""
    try:
        console.log(f"Generating Q-Q plot for {variable} with {distribution} distribution")

        data = state.df[variable].dropna()

        fig, ax = plt.subplots(figsize=(8, 5))

        if distribution == 'norm':
            stats.probplot(data, dist="norm", plot=ax)
            dist_name = "Normal"
        elif distribution == 't':
            stats.probplot(data, dist="t", sparams=(10,), plot=ax)
            dist_name = "t-distribution"
        elif distribution == 'uniform':
            stats.probplot(data, dist="uniform", plot=ax)
            dist_name = "Uniform"
        elif distribution == 'gamma':
            stats.probplot(data, dist="gamma", sparams=(2,), plot=ax)
            dist_name = "Gamma"
        elif distribution == 'exponential':
            stats.probplot(data, dist="expon", plot=ax)
            dist_name = "Exponential"
        else:
            stats.probplot(data, dist="norm", plot=ax)
            dist_name = "Normal"

        ax.set_title(f'Q-Q Plot - {variable} vs {dist_name}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        display(fig, target='qq-plot')
        plt.close(fig)

        console.log("Q-Q plot generated successfully")

    except Exception as e:
        console.error(f"Error generating Q-Q plot: {str(e)}")
        import traceback
        console.error(traceback.format_exc())

# ========== Bivariate Analysis Functions ==========
def show_scatterplot_status(message):
    """Display status message in the scatterplot section"""
    status_elem = document.getElementById('scatterplot-status')
    if status_elem:
        status_elem.innerHTML = message
        status_elem.style.display = 'block'

def hide_scatterplot_status():
    """Hide status message in the scatterplot section"""
    status_elem = document.getElementById('scatterplot-status')
    if status_elem:
        status_elem.style.display = 'none'

def generate_scatterplot(dv, iv, show_linear, show_lowess, show_poly, poly_degree):
    """Generate scatterplot with optional overlays"""
    try:
        df_plot = state.df[[dv, iv]].dropna()
        x = df_plot[iv].values
        y = df_plot[dv].values
        n_points = len(x)

        console.log(f"Bivariate Analysis: Processing {n_points} data points")

        # Show status for large datasets
        if n_points > 10000:
            show_scatterplot_status(f"Processing {n_points:,} data points. Large dataset detected - applying optimizations for better performance...")

        fig, ax = plt.subplots(figsize=(10, 6))

        # Optimize scatterplot rendering for large datasets
        if n_points > 50000:
            # Very large datasets: Aggressive downsampling for visualization
            max_scatter_points = 5000
            sample_indices = np.random.choice(n_points, max_scatter_points, replace=False)
            x_scatter = x[sample_indices]
            y_scatter = y[sample_indices]
            console.log(f"Scatterplot: Heavily downsampled to {max_scatter_points} points (from {n_points})")
            ax.scatter(x_scatter, y_scatter, alpha=0.4, s=20, color='#3b82f6', edgecolors='none')
        elif n_points > 20000:
            # Large datasets: Moderate downsampling
            max_scatter_points = 8000
            sample_indices = np.random.choice(n_points, max_scatter_points, replace=False)
            x_scatter = x[sample_indices]
            y_scatter = y[sample_indices]
            console.log(f"Scatterplot: Downsampled to {max_scatter_points} points (from {n_points})")
            ax.scatter(x_scatter, y_scatter, alpha=0.5, s=25, color='#3b82f6', edgecolors='white', linewidth=0.2)
        elif n_points > 10000:
            # Medium-large datasets: Light downsampling
            max_scatter_points = min(10000, n_points)
            sample_indices = np.random.choice(n_points, max_scatter_points, replace=False)
            x_scatter = x[sample_indices]
            y_scatter = y[sample_indices]
            console.log(f"Scatterplot: Lightly downsampled to {max_scatter_points} points (from {n_points})")
            ax.scatter(x_scatter, y_scatter, alpha=0.5, s=30, color='#3b82f6', edgecolors='white', linewidth=0.3)
        else:
            # Small datasets: Use all points
            ax.scatter(x, y, alpha=0.6, s=50, color='#3b82f6', edgecolors='white', linewidth=0.5)

        # Linear regression line (always uses full dataset)
        if show_linear:
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            ax.plot(x, p(x), color='#dc2626', linewidth=2, label='Linear Regression', linestyle='--')

        # LOWESS smoothing with intelligent downsampling for large datasets
        if show_lowess:
            from statsmodels.nonparametric.smoothers_lowess import lowess

            # Performance optimization with aggressive downsampling for browser constraints
            # PyScript/Pyodide has strict memory limits compared to native Python
            if n_points > 50000:
                # Very large datasets: Use minimal sampling to prevent crashes
                max_lowess_points = 1000
                lowess_frac = 0.05
                console.log(f"LOWESS: Very large dataset ({n_points} points) - using aggressive optimization")
                show_scatterplot_status(f"⚠️ Large dataset ({n_points:,} points): Using aggressive LOWESS optimization (1,000 sampled points, frac=0.05) to prevent browser crash...")
            elif n_points > 20000:
                # Large datasets: Moderate sampling
                max_lowess_points = 1500
                lowess_frac = 0.08
                console.log(f"LOWESS: Large dataset ({n_points} points) - using moderate optimization")
                show_scatterplot_status(f"Computing LOWESS for {n_points:,} points (using 1,500 sampled points, frac=0.08)...")
            elif n_points > 10000:
                # Medium-large datasets: Conservative sampling
                max_lowess_points = 2500
                lowess_frac = 0.10
                console.log(f"LOWESS: Medium-large dataset ({n_points} points) - using light optimization")
                show_scatterplot_status(f"Computing LOWESS for {n_points:,} points (using 2,500 sampled points, frac=0.10)...")
            elif n_points > 5000:
                # Medium datasets: Minimal sampling
                max_lowess_points = 3500
                lowess_frac = 0.15
                console.log(f"LOWESS: Medium dataset ({n_points} points) - using minimal optimization")
            else:
                # Small datasets: Use all points with standard fraction
                max_lowess_points = n_points
                lowess_frac = 0.3
                console.log(f"LOWESS: Using all {n_points} points with frac={lowess_frac}")

            # Apply sampling if needed
            if n_points > max_lowess_points:
                # Sort by x to ensure even coverage across the range
                sort_idx = np.argsort(x)
                x_sorted = x[sort_idx]
                y_sorted = y[sort_idx]
                # Use regular interval sampling for better representation
                sample_indices = np.linspace(0, n_points - 1, max_lowess_points, dtype=int)
                x_sample = x_sorted[sample_indices]
                y_sample = y_sorted[sample_indices]

                console.log(f"LOWESS: Processing {len(x_sample)} sampled points (from {n_points} total) with frac={lowess_frac:.3f}")
                lowess_result = lowess(y_sample, x_sample, frac=lowess_frac)
            else:
                # Use full dataset
                lowess_result = lowess(y, x, frac=lowess_frac)

            ax.plot(lowess_result[:, 0], lowess_result[:, 1], color='#10b981', linewidth=2, label='LOWESS', linestyle='-.')

        # Polynomial regression (uses full dataset)
        if show_poly:
            poly_features = PolynomialFeatures(degree=poly_degree)
            x_poly = poly_features.fit_transform(x.reshape(-1, 1))
            poly_model = LinearRegression()
            poly_model.fit(x_poly, y)

            x_sorted = np.sort(x)
            x_sorted_poly = poly_features.transform(x_sorted.reshape(-1, 1))
            y_poly_pred = poly_model.predict(x_sorted_poly)

            ax.plot(x_sorted, y_poly_pred, color='#f59e0b', linewidth=2,
                   label=f'Polynomial (degree {poly_degree})', linestyle=':')

        ax.set_xlabel(iv, fontsize=12)
        ax.set_ylabel(dv, fontsize=12)
        ax.set_title(f'{dv} vs {iv}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

        if show_linear or show_lowess or show_poly:
            ax.legend(fontsize=10)

        display(fig, target='scatterplot-container')
        plt.close(fig)

        # Hide status message after successful completion
        hide_scatterplot_status()

    except Exception as e:
        console.error(f"Error generating scatterplot: {str(e)}")
        window.appFunctions.showError(f"Error generating scatterplot: {str(e)}")
        hide_scatterplot_status()

# ========== Correlation Analysis Functions ==========
def calculate_correlations():
    """Calculate Pearson, Spearman, and Kendall correlations"""
    try:
        dv = window.appState.selectedDV
        ivs = list(window.appState.selectedIVs)

        results = []

        for iv in ivs:
            df_corr = state.df[[dv, iv]].dropna()

            # Pearson correlation
            pearson_r, pearson_p = stats.pearsonr(df_corr[dv], df_corr[iv])
            pearson_ci = calculate_correlation_ci(pearson_r, len(df_corr))

            # Spearman correlation
            spearman_r, spearman_p = stats.spearmanr(df_corr[dv], df_corr[iv])
            spearman_ci = calculate_correlation_ci(spearman_r, len(df_corr))

            # Kendall's Tau
            kendall_tau, kendall_p = stats.kendalltau(df_corr[dv], df_corr[iv])
            kendall_ci = calculate_correlation_ci(kendall_tau, len(df_corr))

            results.append({
                'IV': iv,
                'Pearson_r': round(pearson_r, 4),
                'Pearson_p': round(pearson_p, 4),
                'Pearson_CI': f"[{pearson_ci[0]:.4f}, {pearson_ci[1]:.4f}]",
                'Spearman_r': round(spearman_r, 4),
                'Spearman_p': round(spearman_p, 4),
                'Spearman_CI': f"[{spearman_ci[0]:.4f}, {spearman_ci[1]:.4f}]",
                'Kendall_tau': round(kendall_tau, 4),
                'Kendall_p': round(kendall_p, 4),
                'Kendall_CI': f"[{kendall_ci[0]:.4f}, {kendall_ci[1]:.4f}]",
                'N': len(df_corr)
            })

        state.correlation_results = results

        # Generate heatmaps
        generate_correlation_heatmaps(results, ivs)

        # Generate table
        generate_correlation_table(results)

    except Exception as e:
        console.error(f"Error calculating correlations: {str(e)}")
        window.appFunctions.showError(f"Error calculating correlations: {str(e)}")

def calculate_correlation_ci(r, n, confidence=0.95):
    """Calculate confidence interval for correlation coefficient using Fisher Z-transformation"""
    z = np.arctanh(r)
    se = 1 / np.sqrt(n - 3)
    z_crit = stats.norm.ppf((1 + confidence) / 2)
    ci_lower = np.tanh(z - z_crit * se)
    ci_upper = np.tanh(z + z_crit * se)
    return (ci_lower, ci_upper)

def generate_correlation_heatmaps(results, ivs):
    """Generate correlation heatmaps"""
    try:
        # Prepare data for heatmaps
        pearson_values = [r['Pearson_r'] for r in results]
        spearman_values = [r['Spearman_r'] for r in results]
        kendall_values = [r['Kendall_tau'] for r in results]

        # Pearson heatmap
        fig, ax = plt.subplots(figsize=(10, 6))
        data_matrix = np.array([pearson_values]).T
        sns.heatmap(data_matrix, annot=True, fmt='.4f', cmap='coolwarm', center=0,
                   yticklabels=ivs, xticklabels=[window.appState.selectedDV],
                   vmin=-1, vmax=1, ax=ax, cbar_kws={'label': 'Correlation'})
        ax.set_title('Pearson Correlation', fontsize=14, fontweight='bold')
        display(fig, target='pearson-heatmap')
        plt.close(fig)

        # Spearman heatmap
        fig, ax = plt.subplots(figsize=(10, 6))
        data_matrix = np.array([spearman_values]).T
        sns.heatmap(data_matrix, annot=True, fmt='.4f', cmap='coolwarm', center=0,
                   yticklabels=ivs, xticklabels=[window.appState.selectedDV],
                   vmin=-1, vmax=1, ax=ax, cbar_kws={'label': 'Correlation'})
        ax.set_title('Spearman Correlation', fontsize=14, fontweight='bold')
        display(fig, target='spearman-heatmap')
        plt.close(fig)

        # Kendall heatmap
        fig, ax = plt.subplots(figsize=(10, 6))
        data_matrix = np.array([kendall_values]).T
        sns.heatmap(data_matrix, annot=True, fmt='.4f', cmap='coolwarm', center=0,
                   yticklabels=ivs, xticklabels=[window.appState.selectedDV],
                   vmin=-1, vmax=1, ax=ax, cbar_kws={'label': 'Correlation'})
        ax.set_title("Kendall's Tau", fontsize=14, fontweight='bold')
        display(fig, target='kendall-heatmap')
        plt.close(fig)

    except Exception as e:
        console.error(f"Error generating heatmaps: {str(e)}")

def generate_correlation_table(results):
    """Generate correlation results table"""
    try:
        table_html = """
        <table>
            <thead>
                <tr>
                    <th>Independent Variable</th>
                    <th>Pearson r</th>
                    <th>p-value</th>
                    <th>95% CI</th>
                    <th>Spearman ρ</th>
                    <th>p-value</th>
                    <th>95% CI</th>
                    <th>Kendall τ</th>
                    <th>p-value</th>
                    <th>95% CI</th>
                    <th>N</th>
                </tr>
            </thead>
            <tbody>
        """

        for r in results:
            table_html += f"""
            <tr>
                <td>{r['IV']}</td>
                <td>{r['Pearson_r']}</td>
                <td>{r['Pearson_p']}</td>
                <td>{r['Pearson_CI']}</td>
                <td>{r['Spearman_r']}</td>
                <td>{r['Spearman_p']}</td>
                <td>{r['Spearman_CI']}</td>
                <td>{r['Kendall_tau']}</td>
                <td>{r['Kendall_p']}</td>
                <td>{r['Kendall_CI']}</td>
                <td>{r['N']}</td>
            </tr>
            """

        table_html += "</tbody></table>"
        document.getElementById('correlation-table-container').innerHTML = table_html

    except Exception as e:
        console.error(f"Error generating correlation table: {str(e)}")

# ========== Plot Display Helper ==========
def display(fig, target):
    """Display matplotlib figure in specified HTML element"""
    try:
        # Convert figure to base64 PNG image
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        buf.close()

        # Get target element and insert image
        target_elem = document.getElementById(target)
        target_elem.innerHTML = f'<img src="data:image/png;base64,{img_base64}" style="max-width: 100%; height: auto;" />'

        console.log(f"Plot displayed successfully in {target}")

    except Exception as e:
        console.error(f"Error displaying plot in {target}: {str(e)}")
        import traceback
        console.error(traceback.format_exc())

# ========== Download Functions ==========
def download_plot_file(plot_id, format_type):
    """Download plot as PNG or SVG"""
    try:
        # This would require additional implementation for canvas to image conversion
        # For now, show a placeholder message
        window.appFunctions.showSuccess(f"Download functionality for {format_type} format")
    except Exception as e:
        console.error(f"Error downloading plot: {str(e)}")

def generate_correlation_pdf():
    """Generate and download correlation results report as HTML"""
    try:
        console.log("Generating correlation report...")

        if not state.correlation_results:
            window.appFunctions.showError("No correlation results to export")
            return

        dv = window.appState.selectedDV
        ivs = list(window.appState.selectedIVs)

        # Create HTML report
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Correlation Analysis Report</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 40px;
            background-color: #f8f9fa;
        }}
        .header {{
            background-color: #2563eb;
            color: white;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 30px;
        }}
        h1 {{
            margin: 0;
            font-size: 24px;
        }}
        .subtitle {{
            margin-top: 5px;
            opacity: 0.9;
        }}
        .section {{
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h2 {{
            color: #2563eb;
            border-bottom: 2px solid #2563eb;
            padding-bottom: 10px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #f1f5f9;
            font-weight: bold;
            color: #1e293b;
        }}
        tr:hover {{
            background-color: #f8fafc;
        }}
        .metric {{
            display: inline-block;
            margin-right: 20px;
            padding: 10px 15px;
            background-color: #f1f5f9;
            border-radius: 6px;
        }}
        .metric-label {{
            font-size: 12px;
            color: #64748b;
            display: block;
        }}
        .metric-value {{
            font-size: 18px;
            font-weight: bold;
            color: #1e293b;
        }}
        .footer {{
            text-align: center;
            color: #64748b;
            margin-top: 30px;
            padding: 20px;
        }}
        .sig {{
            font-weight: bold;
            color: #10b981;
        }}
        .not-sig {{
            color: #94a3b8;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>📊 Correlation Analysis Report</h1>
        <div class="subtitle">Dependent Variable: {dv}</div>
        <div class="subtitle">Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
    </div>

    <div class="section">
        <h2>Analysis Summary</h2>
        <div class="metric">
            <span class="metric-label">Dependent Variable</span>
            <span class="metric-value">{dv}</span>
        </div>
        <div class="metric">
            <span class="metric-label">Independent Variables</span>
            <span class="metric-value">{len(ivs)}</span>
        </div>
        <div class="metric">
            <span class="metric-label">Total Observations</span>
            <span class="metric-value">{len(state.df)}</span>
        </div>
    </div>

    <div class="section">
        <h2>Correlation Results</h2>
        <table>
            <thead>
                <tr>
                    <th>Independent Variable</th>
                    <th>Pearson r</th>
                    <th>p-value</th>
                    <th>95% CI</th>
                    <th>Spearman ρ</th>
                    <th>p-value</th>
                    <th>95% CI</th>
                    <th>Kendall τ</th>
                    <th>p-value</th>
                    <th>95% CI</th>
                    <th>N</th>
                </tr>
            </thead>
            <tbody>
"""

        # Add correlation results
        for r in state.correlation_results:
            pearson_sig = "sig" if r['Pearson_p'] < 0.05 else "not-sig"
            spearman_sig = "sig" if r['Spearman_p'] < 0.05 else "not-sig"
            kendall_sig = "sig" if r['Kendall_p'] < 0.05 else "not-sig"

            html_content += f"""
                <tr>
                    <td><strong>{r['IV']}</strong></td>
                    <td>{r['Pearson_r']}</td>
                    <td class="{pearson_sig}">{r['Pearson_p']}</td>
                    <td>{r['Pearson_CI']}</td>
                    <td>{r['Spearman_r']}</td>
                    <td class="{spearman_sig}">{r['Spearman_p']}</td>
                    <td>{r['Spearman_CI']}</td>
                    <td>{r['Kendall_tau']}</td>
                    <td class="{kendall_sig}">{r['Kendall_p']}</td>
                    <td>{r['Kendall_CI']}</td>
                    <td>{r['N']}</td>
                </tr>
"""

        html_content += """
            </tbody>
        </table>
    </div>

    <div class="section">
        <h2>Interpretation Guide</h2>
        <ul>
            <li><strong>Pearson Correlation (r):</strong> Measures linear relationship between variables. Range: -1 to +1.</li>
            <li><strong>Spearman Correlation (ρ):</strong> Measures monotonic relationship using ranks. Less sensitive to outliers.</li>
            <li><strong>Kendall's Tau (τ):</strong> Another rank-based correlation, more robust for small samples.</li>
            <li><strong>p-value:</strong> Statistical significance. p &lt; 0.05 indicates significant correlation.</li>
            <li><strong>95% CI:</strong> Confidence interval for the correlation coefficient.</li>
        </ul>
    </div>

    <div class="footer">
        <p>Generated by PyScript Statistical Data Analysis Application</p>
        <p>All data processed client-side in your browser for complete privacy</p>
    </div>
</body>
</html>
"""

        # Create blob and trigger download
        from js import Blob, document, window as js_window
        import js

        # Create a Blob from the HTML content
        blob = Blob.new([html_content], {"type": "text/html"})

        # Create download link
        url = js_window.URL.createObjectURL(blob)
        a = document.createElement('a')
        a.href = url
        a.download = f'correlation_report_{dv}_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}.html'
        a.style.display = 'none'
        document.body.appendChild(a)
        a.click()
        document.body.removeChild(a)
        js_window.URL.revokeObjectURL(url)

        console.log("Report downloaded successfully")
        window.appFunctions.showSuccess("Correlation report downloaded successfully!")

    except Exception as e:
        console.error(f"Error generating report: {str(e)}")
        import traceback
        console.error(traceback.format_exc())
        window.appFunctions.showError(f"Error generating report: {str(e)}")

# ========== Expose Python Functions to JavaScript ==========
# These functions can be called from JavaScript
try:
    window.processUploadedFile = process_uploaded_file
    window.checkDataQuality = check_data_quality
    window.handleMissingData = handle_missing_data
    window.detectOutliersInData = detect_outliers_in_data
    window.applyOutlierMethods = apply_outlier_methods
    window.generateUnivariatePlots = generate_univariate_plots
    window.updateQQPlotDistribution = generate_qq_plot
    window.generateScatterplot = generate_scatterplot
    window.calculateCorrelations = calculate_correlations
    window.downloadPlotFile = download_plot_file
    window.generateCorrelationPDF = generate_correlation_pdf

    console.log("✓ Analysis module loaded successfully")
    console.log("✓ All Python functions exposed to JavaScript")
    console.log("✓ Available functions:", [
        "processUploadedFile",
        "checkDataQuality",
        "handleMissingData",
        "detectOutliersInData",
        "applyOutlierMethods",
        "generateUnivariatePlots",
        "updateQQPlotDistribution",
        "generateScatterplot",
        "calculateCorrelations"
    ])

    # Signal that Python is ready
    window.pythonReady = True

except Exception as e:
    console.error(f"Error exposing Python functions: {str(e)}")
    import traceback
    console.error(traceback.format_exc())
