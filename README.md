# PyScript Statistical Data Analysis Application

A comprehensive web-based statistical data analysis application that processes all data client-side in the browser using **PyScript/Pyodide**. This ensures complete privacy - no data is ever sent to a server.

## Overview

This application provides a complete workflow for statistical data analysis, from data upload through correlation analysis, all running in the user's browser. It's designed for researchers, data analysts, and students who need privacy-preserving statistical analysis tools.

## Key Features

- **Privacy-First Design**: All data processing happens in the browser - no server uploads
- **No Installation Required**: Access via web browser, no Python installation needed
- **Comprehensive Analysis Pipeline**:
  - Data quality checks and descriptive statistics
  - Flexible missing data handling (delete, impute, or manual fix)
  - Multiple outlier detection methods (IQR, Z-score, Modified Z-score)
  - Univariate analysis with KDE, box plots, and Q-Q plots
  - Bivariate scatterplots with regression overlays
  - Correlation analysis (Pearson, Spearman, Kendall)
- **Interactive Visualizations**: All plots are interactive and downloadable
- **Professional UI**: Clean, light-mode interface with step-by-step guidance

## Quick Start

### Local Development

1. **Navigate to the app directory**:
   ```bash
   cd app
   ```

2. **Start a local web server**:
   ```bash
   python -m http.server 8000
   ```

3. **Open your browser**:
   ```
   http://localhost:8000
   ```

4. **Try the sample data**:
   - Use the included `sample_data.csv` file to test the application
   - The sample contains sales data with multiple variables for analysis

### Deploy to Production

Deploy the `app` folder to any static hosting service:
- GitHub Pages
- Netlify
- Vercel
- AWS S3 Static Website
- Any web server

**No server-side configuration needed** - it's purely client-side!

## File Structure

```
edaset/
├── app/
│   ├── index.html           # Main application interface
│   ├── styles.css           # Light mode styling
│   ├── script.js            # UI logic and state management
│   ├── analysis.py          # Python statistical analysis
│   ├── pyscript.toml        # PyScript configuration
│   ├── sample_data.csv      # Sample dataset for testing
│   └── README.md            # Detailed usage instructions
├── .gitignore
└── README.md                # This file
```

## Technical Stack

- **Frontend**: HTML5, CSS3, JavaScript (ES6+)
- **Python Runtime**: PyScript 2024.1.1 / Pyodide 0.24.1
- **Libraries**: pandas, numpy, scipy, matplotlib, seaborn, scikit-learn, statsmodels
- **Deployment**: Static hosting (no backend required)

## Requirements

- Modern web browser (Chrome 90+, Firefox 88+, Safari 14+, Edge 90+)
- JavaScript enabled
- Minimum 4GB RAM recommended
- Internet connection (for initial library loading)

## Data Requirements

- **Formats**: CSV, XLS, XLSX
- **Size**: Maximum 50 MB
- **Variables**: Numeric only (categorical variables are filtered out)

## Usage Workflow

1. **Initialize**: Click "Start Analysis" to load Python libraries
2. **Upload**: Drag and drop or select your data file
3. **Select Variables**: Choose 1 dependent variable and 1+ independent variables
4. **Data Quality**: Review statistics and handle missing data
5. **Outliers**: Detect and handle outliers per variable
6. **Univariate**: Analyze distributions with KDE, box, and Q-Q plots
7. **Bivariate**: View scatterplots with regression overlays
8. **Correlations**: Calculate and visualize correlations

## Features in Detail

### Data Quality Check
- Automatic detection of missing/invalid values
- Descriptive statistics for all selected variables
- Multiple handling options: delete, impute (mean/median/mode), or manual fix

### Outlier Detection
- **IQR Method**: 1.5× interquartile range
- **Z-Score**: |z| > 3
- **Modified Z-Score**: Using Median Absolute Deviation (MAD)
- Per-variable handling: ignore, delete, winsorize, replace, or transform

### Univariate Analysis
- **KDE Plot**: Kernel Density Estimate for distribution visualization
- **Box Plot**: Median, quartiles, and outliers
- **Q-Q Plot**: Compare against Normal, t, Uniform, Gamma, or Exponential distributions

### Bivariate Analysis
- Scatterplots for DV vs each IV
- Overlay options:
  - Linear regression line
  - LOWESS smoothing curve
  - Polynomial regression (degree 2, 3, or 4)

### Correlation Analysis
- **Pearson**: Linear relationships
- **Spearman**: Monotonic relationships
- **Kendall's Tau**: Rank-based correlation
- Results include: coefficient, p-value, 95% CI, sample size
- Heatmap and table views

## Browser Support

| Browser | Minimum Version | Recommended |
|---------|----------------|-------------|
| Chrome  | 90+            | Latest      |
| Firefox | 88+            | Latest      |
| Safari  | 14+            | Latest      |
| Edge    | 90+            | Latest      |

## Performance

- **Optimal**: Datasets with <10,000 rows
- **RAM Usage**: 200-500 MB during analysis
- **Initial Load**: 10-30 seconds (first visit only)
- **Processing**: Most operations complete in 1-5 seconds

## Privacy & Security

- All data processing occurs locally in your browser
- No data transmission to any server
- No cookies, tracking, or analytics
- Data cleared on page refresh
- Perfect for sensitive or confidential data

## Limitations

- Numeric variables only
- Maximum file size: 50 MB
- Requires WebAssembly support
- Large datasets may impact performance on low-end devices

## Troubleshooting

See `app/README.md` for detailed troubleshooting guide covering:
- Library loading issues
- File upload problems
- Browser freezing
- Plot display issues

## Future Enhancements

Potential features for future versions:
- Support for categorical variables
- Multiple regression analysis
- Time series analysis
- Export analysis as Python script
- Save/load analysis sessions
- Additional plot types
- Advanced statistical tests (t-tests, ANOVA)

## Contributing

This project is open for contributions:
- Bug fixes
- Feature enhancements
- Documentation improvements
- Performance optimizations

## License

This application is provided as-is for educational and research purposes.

## Credits

Built with open-source technologies:
- [PyScript](https://pyscript.net/) - Python in the browser
- [Pyodide](https://pyodide.org/) - Python on WebAssembly
- [Matplotlib](https://matplotlib.org/) - Visualization
- [Pandas](https://pandas.pydata.org/) - Data analysis
- [SciPy](https://scipy.org/) - Scientific computing
- [scikit-learn](https://scikit-learn.org/) - Machine learning
- [Seaborn](https://seaborn.pydata.org/) - Statistical visualization
- [statsmodels](https://www.statsmodels.org/) - Statistical modeling

## Support

For detailed usage instructions and troubleshooting, see `app/README.md`.

---

**Built for privacy-conscious data analysis. Your data never leaves your browser.** 🔒
