# Statistical Data Analysis Web Application

A privacy-focused, client-side statistical analysis application that processes all data in the user's browser using PyScript/Pyodide. No data is sent to any server, ensuring complete privacy and security.

## Features

- **100% Client-Side Processing**: All data analysis happens in your browser's RAM
- **Privacy First**: No data is uploaded to any server
- **Comprehensive Statistical Analysis**:
  - Data quality checks and descriptive statistics
  - Missing data detection and handling
  - Outlier detection (IQR, Z-score, Modified Z-score)
  - Univariate analysis (KDE, Box plots, Q-Q plots)
  - Bivariate analysis (Scatterplots with multiple overlays)
  - Correlation analysis (Pearson, Spearman, Kendall's Tau)
- **Interactive Visualizations**: All plots are interactive and downloadable
- **User-Friendly Interface**: Step-by-step guided workflow with light mode design

## Technical Requirements

### Browser Requirements
- Modern web browser (Chrome 90+, Firefox 88+, Safari 14+, Edge 90+)
- Minimum 4GB RAM recommended
- JavaScript enabled

### File Requirements
- **Supported Formats**: CSV, XLS, XLSX
- **Maximum File Size**: 50 MB
- **Data Type**: Numeric variables only (categorical variables are filtered out)

## Installation & Deployment

### Option 1: Local Development Server

1. **Clone or download this repository**

2. **Navigate to the app directory**
   ```bash
   cd app
   ```

3. **Start a local web server**

   Using Python 3:
   ```bash
   python -m http.server 8000
   ```

   Using Node.js (http-server):
   ```bash
   npx http-server -p 8000
   ```

   Using PHP:
   ```bash
   php -S localhost:8000
   ```

4. **Open your browser**
   Navigate to: `http://localhost:8000`

### Option 2: Deploy to Web Hosting

Deploy the entire `app` folder to any static web hosting service:

- **GitHub Pages**: Push to a repository and enable GitHub Pages
- **Netlify**: Drag and drop the app folder
- **Vercel**: Deploy via CLI or web interface
- **AWS S3**: Upload files and enable static website hosting
- **Any web server**: Copy files to your web server's public directory

**Important**: No server-side configuration is needed. The application is purely client-side.

## Usage Guide

### Step 1: Initialization
- Click "Start Analysis" to initialize the Python environment
- Wait for all libraries to load (pandas, numpy, scipy, matplotlib, scikit-learn)
- This takes 10-30 seconds depending on your internet speed

### Step 2: Data Upload
- Click "Click to upload" or drag and drop your data file
- Supported formats: CSV, XLS, XLSX
- Maximum file size: 50 MB
- The application will automatically load and parse your data

### Step 3: Variable Selection
- Select **one Dependent Variable (DV)** from the dropdown
- Select **one or more Independent Variables (IVs)** using checkboxes
- Variables must be numeric (non-numeric columns are automatically filtered)
- Click "Next" when ready

### Step 4: Data Quality Check
- Review dataset summary and descriptive statistics
- Check for missing or invalid data
- Choose missing data handling method:
  - **Delete rows**: Removes any row with missing values (default)
  - **Impute**: Replace missing values with mean/median/mode per variable
  - **Manual fix**: View exact locations of missing data for manual correction

### Step 5: Outlier Detection
- Select detection method:
  - **IQR Method**: Uses 1.5× interquartile range
  - **Z-Score**: Identifies values with |z| > 3
  - **Modified Z-Score**: Uses Median Absolute Deviation (MAD)
- Review outlier summary for each variable
- Choose handling method per variable:
  - Ignore
  - Delete rows
  - Winsorize (cap at bounds)
  - Replace with mean/median
  - Apply log or square root transformation

### Step 6: Univariate Analysis
- Select a variable from the dropdown
- View three plots:
  1. **KDE Plot**: Kernel Density Estimate
  2. **Box Plot**: Shows median, quartiles, and outliers
  3. **Q-Q Plot**: Compare against reference distribution (Normal, t, Uniform, etc.)
- Click any plot to maximize to full screen
- Download plots as PNG or SVG

### Step 7: Bivariate Analysis
- View scatterplots of DV vs each IV
- Navigate between IVs using Previous/Next buttons
- Toggle plot overlays:
  - Linear regression line
  - LOWESS smoothing curve
  - Polynomial regression (degree 2, 3, or 4)
- Download scatterplots as PNG or SVG

### Step 8: Correlation Analysis
- View correlation results in two formats:
  1. **Heatmap View**: Visual representation of correlations
  2. **Table View**: Detailed statistics with p-values and confidence intervals
- Three correlation types calculated:
  - Pearson (linear relationships)
  - Spearman (monotonic relationships)
  - Kendall's Tau (rank-based correlation)
- Download complete correlation report as PDF
- Click "Start New Analysis" to analyze a different dataset

## Application Features

### Progress Indicators
- Loading bars and spinners show progress during:
  - Library initialization
  - Data upload and processing
  - Data cleaning operations
  - Plot generation
  - Correlation computation

### Error Handling
The application provides clear error messages for:
- File size exceeding 50 MB
- Invalid file formats
- Browser RAM limitations
- Computation errors
- Missing or invalid data

### Keyboard Shortcuts
- `Alt + →`: Next step
- `Alt + ←`: Previous step
- `Esc`: Close modal windows

### Data Privacy
- All processing happens locally in your browser
- No data is transmitted to any server
- Data is cleared when you refresh the page
- No cookies or tracking

## File Structure

```
app/
├── index.html          # Main application page with UI structure
├── styles.css          # Light mode styling and responsive design
├── script.js           # Frontend logic and state management
├── analysis.py         # Python statistical analysis functions
├── pyscript.toml       # PyScript/Pyodide configuration
└── README.md           # This file
```

## Troubleshooting

### Issue: Libraries taking too long to load
**Solution**:
- Ensure stable internet connection
- Close other browser tabs to free up memory
- Try a different browser
- Clear browser cache and reload

### Issue: File upload fails
**Solution**:
- Check file size is under 50 MB
- Ensure file format is CSV, XLS, or XLSX
- Verify file contains numeric data
- Try opening the file in Excel to check for corruption

### Issue: Browser freezes during analysis
**Solution**:
- Your dataset may be too large for available RAM
- Close other applications to free up memory
- Try with a smaller subset of your data
- Use a desktop browser instead of mobile

### Issue: Plots not displaying
**Solution**:
- Wait for all libraries to fully load
- Check browser console for errors (F12)
- Ensure JavaScript is enabled
- Try refreshing the page

### Issue: Missing data or outlier options not working
**Solution**:
- Ensure you've selected variables in Step 3
- Wait for data quality check to complete
- Check that your data contains the expected values

## Technical Details

### Libraries Used
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **scipy**: Statistical functions
- **matplotlib**: Plotting and visualization
- **seaborn**: Statistical data visualization
- **scikit-learn**: Machine learning algorithms (regression, preprocessing)
- **statsmodels**: Statistical modeling (LOWESS smoothing)

### Browser Compatibility
The application uses modern web standards:
- ES6+ JavaScript
- CSS Grid and Flexbox
- HTML5 File API
- PyScript/Pyodide for Python execution

### Performance Considerations
- **Dataset size**: Optimal performance with datasets under 10,000 rows
- **RAM usage**: Expect 200-500 MB RAM usage during analysis
- **Processing time**: Most operations complete within 1-5 seconds
- **Initial load**: First-time library loading takes 10-30 seconds

## Limitations

- Only numeric variables are supported (categorical data is excluded)
- Maximum file size: 50 MB
- Requires modern browser with WebAssembly support
- Internet connection required for initial library loading (first visit only)
- Large datasets may cause performance issues on low-end devices

## Future Enhancements

Potential features for future versions:
- Support for categorical variables
- Multiple regression analysis
- Time series analysis
- Export analysis workflow as Python script
- Save/load analysis sessions
- Additional plot types (violin plots, pair plots)
- Advanced statistical tests (t-tests, ANOVA)

## Support & Feedback

For issues, questions, or feature requests:
- Check the Troubleshooting section above
- Review browser console for error messages (F12)
- Ensure your data format matches requirements
- Try with a sample dataset to isolate issues

## License

This application is provided as-is for educational and research purposes.

## Credits

Built with:
- [PyScript](https://pyscript.net/) - Run Python in the browser
- [Pyodide](https://pyodide.org/) - Python runtime for WebAssembly
- [Matplotlib](https://matplotlib.org/) - Visualization library
- [Pandas](https://pandas.pydata.org/) - Data analysis library
- [SciPy](https://scipy.org/) - Scientific computing library

---

**Privacy Notice**: This application processes all data locally in your browser. No data is transmitted to any server. Your data remains completely private and secure.
