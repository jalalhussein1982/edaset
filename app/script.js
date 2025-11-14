/**
 * Statistical Data Analysis Application
 * Client-side JavaScript for UI controls and state management
 */

// ========== Global State Management ==========
const appState = {
    currentStep: 1,
    dataFile: null,
    dataFrame: null,
    columns: [],
    selectedDV: null,
    selectedIVs: [],
    cleanedData: null,
    outlierMethod: 'iqr',
    currentScatterplotIndex: 0,
    correlationView: 'heatmap'
};

// Make appState globally accessible for Python
window.appState = appState;

// ========== Initialization ==========
document.addEventListener('DOMContentLoaded', function() {
    console.log('Application initialized');
    updateNavigationState();

    // Listen for PyScript ready event
    document.addEventListener('py:ready', function() {
        console.log('PyScript is ready!');
        window.pythonReady = true;
    });

    document.addEventListener('py:all-done', function() {
        console.log('All PyScript code loaded!');
        window.pythonReady = true;
    });
});

// ========== Progress Bar Functions ==========
function showProgress(message, percent = 0) {
    const progressContainer = document.getElementById('global-progress');
    const progressFill = document.getElementById('progress-fill');
    const progressText = document.getElementById('progress-text');

    progressContainer.classList.remove('hidden');
    progressFill.style.width = percent + '%';
    progressText.textContent = message;
}

function hideProgress() {
    const progressContainer = document.getElementById('global-progress');
    progressContainer.classList.add('hidden');
}

function updateProgress(percent, message) {
    const progressFill = document.getElementById('progress-fill');
    const progressText = document.getElementById('progress-text');
    progressFill.style.width = percent + '%';
    if (message) progressText.textContent = message;
}

// ========== Step Navigation ==========
function goToStep(stepNumber) {
    // Hide all steps
    const allSteps = document.querySelectorAll('.step-section');
    allSteps.forEach(step => step.classList.remove('active'));

    // Show current step
    const currentStepEl = document.getElementById(`step-${stepNumber}`);
    if (currentStepEl) {
        currentStepEl.classList.add('active');
    }

    // Update sidebar navigation
    const navItems = document.querySelectorAll('#nav-steps li');
    navItems.forEach((item, index) => {
        item.classList.remove('active');
        if (index + 1 === stepNumber) {
            item.classList.add('active');
        }
        if (index + 1 < stepNumber) {
            item.classList.add('completed');
        } else {
            item.classList.remove('completed');
        }
    });

    appState.currentStep = stepNumber;

    // Trigger step-specific actions
    handleStepTransition(stepNumber);

    window.scrollTo(0, 0);
}

function handleStepTransition(stepNumber) {
    // Handle actions when transitioning to specific steps
    switch(stepNumber) {
        case 3:
            // Process uploaded file when entering variable selection step
            if (appState.dataFile) {
                console.log('Step 3: Checking if Python is ready...');
                waitForPythonThenProcess();
            }
            break;
        case 4:
            // Check data quality when entering data quality step
            waitForPythonThenCall('checkDataQuality', window.checkDataQuality);
            break;
        case 5:
            // Detect outliers when entering outlier step
            waitForPythonThenCall('detectOutliersInData', detectOutliers);
            break;
        case 6:
            // Populate univariate variable dropdown
            populateUnivariateDropdown();
            break;
        case 7:
            // Initialize scatterplot
            appState.currentScatterplotIndex = 0;
            setTimeout(() => updateScatterplot(), 100);
            break;
        case 8:
            // Calculate correlations
            waitForPythonThenCall('calculateCorrelations', window.calculateCorrelations);
            break;
    }
}

function waitForPythonThenProcess() {
    // Wait for Python to be ready before processing file
    let attempts = 0;
    const maxAttempts = 50; // 10 seconds max

    const checkAndProcess = () => {
        attempts++;
        console.log(`Attempt ${attempts}: Checking for processUploadedFile...`);

        if (typeof window.processUploadedFile === 'function') {
            console.log('Python ready! Processing file...');
            window.processUploadedFile();
        } else if (attempts < maxAttempts) {
            console.log('Python not ready yet, retrying...');
            setTimeout(checkAndProcess, 200);
        } else {
            console.error('Python failed to load after 10 seconds');
            alert('Python environment is still loading. Please wait a moment and try clicking Next again.');
        }
    };

    checkAndProcess();
}

function waitForPythonThenCall(functionName, functionRef) {
    // Generic function to wait for Python functions
    let attempts = 0;
    const maxAttempts = 50;

    const checkAndCall = () => {
        attempts++;

        if (typeof functionRef === 'function') {
            console.log(`${functionName} is ready, calling it...`);
            setTimeout(() => functionRef(), 100);
        } else if (attempts < maxAttempts) {
            setTimeout(checkAndCall, 200);
        } else {
            console.warn(`${functionName} not available after waiting`);
        }
    };

    checkAndCall();
}

function nextStep() {
    if (validateCurrentStep()) {
        goToStep(appState.currentStep + 1);
    }
}

function previousStep() {
    if (appState.currentStep > 1) {
        goToStep(appState.currentStep - 1);
    }
}

function validateCurrentStep() {
    // Add validation logic for each step
    switch (appState.currentStep) {
        case 2:
            return appState.dataFile !== null;
        case 3:
            return appState.selectedDV !== null && appState.selectedIVs.length > 0;
        default:
            return true;
    }
}

function updateNavigationState() {
    // Enable/disable navigation buttons based on state
    const uploadNextBtn = document.getElementById('upload-next-btn');
    const variableNextBtn = document.getElementById('variable-next-btn');

    if (uploadNextBtn) {
        uploadNextBtn.disabled = appState.dataFile === null;
    }

    if (variableNextBtn) {
        variableNextBtn.disabled = appState.selectedDV === null || appState.selectedIVs.length === 0;
    }
}

// ========== Step 1: Application Start ==========
function startApplication() {
    showProgress('Initializing Python environment...', 10);
    updateLibraryStatus('pandas', 'loading');
    updateLibraryStatus('numpy', 'loading');
    updateLibraryStatus('scipy', 'loading');
    updateLibraryStatus('matplotlib', 'loading');
    updateLibraryStatus('scikit', 'loading');

    // Simulate library loading (PyScript will actually handle this)
    setTimeout(() => {
        updateLibraryStatus('pandas', 'complete');
        updateProgress(30, 'Loading numpy...');
    }, 500);

    setTimeout(() => {
        updateLibraryStatus('numpy', 'complete');
        updateProgress(50, 'Loading scipy...');
    }, 1000);

    setTimeout(() => {
        updateLibraryStatus('scipy', 'complete');
        updateProgress(70, 'Loading matplotlib...');
    }, 1500);

    setTimeout(() => {
        updateLibraryStatus('matplotlib', 'complete');
        updateProgress(90, 'Loading scikit-learn...');
    }, 2000);

    setTimeout(() => {
        updateLibraryStatus('scikit', 'complete');
        updateProgress(100, 'Ready!');
        hideProgress();
        goToStep(2);
    }, 2500);
}

function updateLibraryStatus(libName, status) {
    const libElement = document.getElementById(`lib-${libName}`);
    if (!libElement) return;

    const statusSpan = libElement.querySelector('span');
    statusSpan.className = `status-${status}`;

    switch (status) {
        case 'loading':
            statusSpan.textContent = 'Loading...';
            break;
        case 'complete':
            statusSpan.textContent = '✓ Loaded';
            break;
        case 'error':
            statusSpan.textContent = '✗ Error';
            break;
        default:
            statusSpan.textContent = 'Pending';
    }
}

// ========== Step 2: File Upload ==========
function handleFileUpload(event) {
    const file = event.target.files[0];

    if (!file) return;

    // Validate file size (50 MB = 50 * 1024 * 1024 bytes)
    const maxSize = 50 * 1024 * 1024;
    if (file.size > maxSize) {
        alert('Error: File size exceeds 50 MB limit. Please upload a smaller file.');
        event.target.value = '';
        return;
    }

    // Validate file type
    const validExtensions = ['.csv', '.xls', '.xlsx'];
    const fileName = file.name.toLowerCase();
    const isValidType = validExtensions.some(ext => fileName.endsWith(ext));

    if (!isValidType) {
        alert('Error: Invalid file format. Please upload a CSV or XLS file.');
        event.target.value = '';
        return;
    }

    appState.dataFile = file;

    // Display file info
    const fileInfoDiv = document.getElementById('file-info');
    const fileNameSpan = document.getElementById('file-name');
    const fileSizeSpan = document.getElementById('file-size');

    fileNameSpan.textContent = file.name;
    fileSizeSpan.textContent = formatFileSize(file.size);
    fileInfoDiv.classList.remove('hidden');

    // Enable next button
    document.getElementById('upload-next-btn').disabled = false;

    // Load file with PyScript
    loadFileWithPython(file);
}

function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
}

function loadFileWithPython(file) {
    showProgress('Loading data file...', 0);

    const reader = new FileReader();
    reader.onload = function(e) {
        const fileContent = e.target.result;

        try {
            // Store file content in window for Python access
            window.fileContent = fileContent;
            window.fileName = file.name;

            updateProgress(50, 'File read successfully...');

            // Update progress to 100
            setTimeout(() => {
                updateProgress(100, 'File loaded! Ready to proceed.');
                setTimeout(hideProgress, 1000);
            }, 500);

            console.log('File loaded successfully:', file.name);
        } catch (error) {
            console.error('Error loading file:', error);
            alert('Error loading file. Please try again.');
            hideProgress();
        }
    };

    reader.onerror = function() {
        alert('Error reading file. Please try again.');
        hideProgress();
    };

    // Progress simulation
    setTimeout(() => updateProgress(25, 'Reading file...'), 200);

    if (file.name.endsWith('.csv')) {
        reader.readAsText(file);
    } else {
        reader.readAsArrayBuffer(file);
    }
}

// ========== Step 3: Variable Selection ==========
function populateVariableSelections(columns) {
    appState.columns = columns;

    // Populate DV select
    const dvSelect = document.getElementById('dv-select');
    dvSelect.innerHTML = '<option value="">-- Select Dependent Variable --</option>';

    columns.forEach(col => {
        const option = document.createElement('option');
        option.value = col;
        option.textContent = col;
        dvSelect.appendChild(option);
    });

    // Populate IV checkboxes
    const ivCheckboxes = document.getElementById('iv-checkboxes');
    ivCheckboxes.innerHTML = '';

    columns.forEach(col => {
        const label = document.createElement('label');
        const checkbox = document.createElement('input');
        checkbox.type = 'checkbox';
        checkbox.value = col;
        checkbox.onchange = updateVariableSelection;

        label.appendChild(checkbox);
        label.appendChild(document.createTextNode(col));
        ivCheckboxes.appendChild(label);
    });

    // Add DV change listener
    dvSelect.onchange = updateVariableSelection;
}

function updateVariableSelection() {
    const dvSelect = document.getElementById('dv-select');
    const ivCheckboxes = document.querySelectorAll('#iv-checkboxes input[type="checkbox"]');

    appState.selectedDV = dvSelect.value || null;
    appState.selectedIVs = [];

    ivCheckboxes.forEach(checkbox => {
        // Disable if same as DV
        if (checkbox.value === appState.selectedDV) {
            checkbox.checked = false;
            checkbox.disabled = true;
        } else {
            checkbox.disabled = false;
            if (checkbox.checked) {
                appState.selectedIVs.push(checkbox.value);
            }
        }
    });

    // Update summary
    const selectionSummary = document.getElementById('selection-summary');
    const selectedDVSpan = document.getElementById('selected-dv');
    const selectedIVsSpan = document.getElementById('selected-ivs');

    if (appState.selectedDV || appState.selectedIVs.length > 0) {
        selectionSummary.classList.remove('hidden');
        selectedDVSpan.textContent = appState.selectedDV || 'None';
        selectedIVsSpan.textContent = appState.selectedIVs.length > 0
            ? appState.selectedIVs.join(', ')
            : 'None';
    }

    // Enable/disable next button
    const variableNextBtn = document.getElementById('variable-next-btn');
    variableNextBtn.disabled = !appState.selectedDV || appState.selectedIVs.length === 0;
}

// ========== Step 4: Missing Data Handling ==========
function applyMissingDataMethod() {
    const selectedMethod = document.querySelector('input[name="missing-method"]:checked').value;

    showProgress('Processing missing data...', 50);

    // Call Python function to handle missing data
    if (typeof handleMissingData !== 'undefined') {
        handleMissingData(selectedMethod);
    }

    setTimeout(() => {
        hideProgress();
        // If manual fix was selected, show instruction
        if (selectedMethod === 'manual') {
            alert('Please fix the data manually and restart the application.');
        } else {
            // Continue to next step
            nextStep();
        }
    }, 1000);
}

// Listen for missing method radio changes
document.addEventListener('DOMContentLoaded', function() {
    const missingRadios = document.querySelectorAll('input[name="missing-method"]');
    missingRadios.forEach(radio => {
        radio.addEventListener('change', function() {
            const imputeOptions = document.getElementById('impute-options');
            if (this.value === 'impute') {
                imputeOptions.classList.remove('hidden');
            } else {
                imputeOptions.classList.add('hidden');
            }
        });
    });
});

// ========== Step 5: Outlier Detection ==========
function detectOutliers() {
    const selectedMethod = document.querySelector('input[name="outlier-method"]:checked').value;
    appState.outlierMethod = selectedMethod;

    showProgress('Detecting outliers...', 50);

    // Call Python function to detect outliers
    if (typeof detectOutliersInData !== 'undefined') {
        detectOutliersInData(selectedMethod);
    }

    setTimeout(hideProgress, 1000);
}

function applyOutlierHandling() {
    showProgress('Applying outlier handling...', 50);

    // Collect outlier handling methods for each variable
    const outlierMethods = {};
    const selects = document.querySelectorAll('[id^="outlier-method-"]');

    selects.forEach(select => {
        const varName = select.id.replace('outlier-method-', '');
        outlierMethods[varName] = select.value;
    });

    // Call Python function to apply outlier handling
    if (typeof applyOutlierMethods !== 'undefined') {
        applyOutlierMethods(outlierMethods);
    }

    setTimeout(() => {
        hideProgress();
        nextStep();
    }, 1000);
}

// ========== Step 6: Univariate Analysis ==========
function showUnivariatePlots() {
    const selectedVariable = document.getElementById('univariate-variable').value;

    if (!selectedVariable) return;

    showProgress('Generating univariate plots...', 30);

    // Call Python function to generate plots
    if (typeof generateUnivariatePlots !== 'undefined') {
        generateUnivariatePlots(selectedVariable);
    }

    setTimeout(hideProgress, 1500);
}

function updateQQPlot() {
    const selectedVariable = document.getElementById('univariate-variable').value;
    const distribution = document.getElementById('qq-distribution').value;

    if (!selectedVariable) return;

    showProgress('Updating Q-Q plot...', 50);

    // Call Python function to update Q-Q plot
    if (typeof updateQQPlotDistribution !== 'undefined') {
        updateQQPlotDistribution(selectedVariable, distribution);
    }

    setTimeout(hideProgress, 500);
}

function populateUnivariateDropdown() {
    // Populate univariate variable dropdown with all selected variables (DV + IVs)
    const dropdown = document.getElementById('univariate-variable');
    dropdown.innerHTML = '<option value="">-- Select Variable --</option>';

    const allVariables = [appState.selectedDV, ...appState.selectedIVs];

    allVariables.forEach(varName => {
        const option = document.createElement('option');
        option.value = varName;
        option.textContent = varName;
        dropdown.appendChild(option);
    });

    console.log('Univariate dropdown populated with:', allVariables);
}

// ========== Step 7: Bivariate Analysis ==========
function updateScatterplot() {
    if (appState.selectedIVs.length === 0) return;

    const currentIV = appState.selectedIVs[appState.currentScatterplotIndex];

    // Get overlay options
    const showLinear = document.getElementById('overlay-linear').checked;
    const showLowess = document.getElementById('overlay-lowess').checked;
    const showPoly = document.getElementById('overlay-poly').checked;
    const polyDegree = parseInt(document.getElementById('poly-degree').value);

    showProgress('Generating scatterplot...', 50);

    // Update title
    document.getElementById('scatter-title').textContent = `${appState.selectedDV} vs ${currentIV}`;

    // Call Python function to generate scatterplot
    if (typeof generateScatterplot !== 'undefined') {
        generateScatterplot(appState.selectedDV, currentIV, showLinear, showLowess, showPoly, polyDegree);
    }

    // Update button states
    document.getElementById('prev-scatter-btn').disabled = appState.currentScatterplotIndex === 0;
    document.getElementById('next-scatter-btn').disabled = appState.currentScatterplotIndex === appState.selectedIVs.length - 1;

    setTimeout(hideProgress, 1000);
}

function previousScatterplot() {
    if (appState.currentScatterplotIndex > 0) {
        appState.currentScatterplotIndex--;
        updateScatterplot();
    }
}

function nextScatterplot() {
    if (appState.currentScatterplotIndex < appState.selectedIVs.length - 1) {
        appState.currentScatterplotIndex++;
        updateScatterplot();
    }
}

// ========== Step 8: Correlation Analysis ==========
function toggleCorrelationView(view) {
    appState.correlationView = view;

    const heatmapView = document.getElementById('correlation-heatmap');
    const tableView = document.getElementById('correlation-table');
    const heatmapBtn = document.getElementById('heatmap-btn');
    const tableBtn = document.getElementById('table-btn');

    if (view === 'heatmap') {
        heatmapView.classList.remove('hidden');
        tableView.classList.add('hidden');
        heatmapBtn.classList.add('active');
        tableBtn.classList.remove('active');
    } else {
        heatmapView.classList.add('hidden');
        tableView.classList.remove('hidden');
        heatmapBtn.classList.remove('active');
        tableBtn.classList.add('active');
    }
}

function downloadCorrelationResults() {
    showProgress('Generating PDF report...', 50);

    // Call Python function to generate PDF
    if (typeof generateCorrelationPDF !== 'undefined') {
        generateCorrelationPDF();
    }

    setTimeout(hideProgress, 2000);
}

// ========== Plot Functions ==========
function maximizePlot(plotId) {
    const modal = document.getElementById('plot-modal');
    const modalContainer = document.getElementById('modal-plot-container');
    const plotElement = document.getElementById(plotId);

    if (!plotElement) return;

    // Clone plot to modal
    const clonedPlot = plotElement.cloneNode(true);
    modalContainer.innerHTML = '';
    modalContainer.appendChild(clonedPlot);

    modal.classList.remove('hidden');
    modal.classList.add('active');
}

function closeModal() {
    const modal = document.getElementById('plot-modal');
    modal.classList.remove('active');
    modal.classList.add('hidden');
}

function downloadPlot(plotId, format) {
    showProgress(`Downloading ${format.toUpperCase()}...`, 50);

    // Call Python function to download plot
    if (typeof downloadPlotFile !== 'undefined') {
        downloadPlotFile(plotId, format);
    }

    setTimeout(hideProgress, 1000);
}

// ========== Missing Data Modal ==========
function showMissingDataModal(data) {
    const modal = document.getElementById('missing-data-modal');
    const tableContainer = document.getElementById('missing-data-table');

    // Create table
    let tableHTML = `
        <table>
            <thead>
                <tr>
                    <th>Row Number</th>
                    <th>Column Name</th>
                    <th>Variable Name</th>
                    <th>Error Type</th>
                </tr>
            </thead>
            <tbody>
    `;

    data.forEach(item => {
        tableHTML += `
            <tr>
                <td>${item.row}</td>
                <td>${item.column}</td>
                <td>${item.variable}</td>
                <td>${item.errorType}</td>
            </tr>
        `;
    });

    tableHTML += '</tbody></table>';
    tableContainer.innerHTML = tableHTML;

    modal.classList.remove('hidden');
    modal.classList.add('active');
}

function closeMissingDataModal() {
    const modal = document.getElementById('missing-data-modal');
    modal.classList.remove('active');
    modal.classList.add('hidden');
}

// Close modals on outside click
window.onclick = function(event) {
    const plotModal = document.getElementById('plot-modal');
    const missingModal = document.getElementById('missing-data-modal');

    if (event.target === plotModal) {
        closeModal();
    }
    if (event.target === missingModal) {
        closeMissingDataModal();
    }
};

// ========== Drag and Drop Support ==========
const uploadArea = document.getElementById('upload-area');

if (uploadArea) {
    uploadArea.addEventListener('dragover', function(e) {
        e.preventDefault();
        e.stopPropagation();
        this.style.borderColor = 'var(--primary-color)';
        this.style.backgroundColor = '#eff6ff';
    });

    uploadArea.addEventListener('dragleave', function(e) {
        e.preventDefault();
        e.stopPropagation();
        this.style.borderColor = 'var(--border)';
        this.style.backgroundColor = 'var(--background)';
    });

    uploadArea.addEventListener('drop', function(e) {
        e.preventDefault();
        e.stopPropagation();
        this.style.borderColor = 'var(--border)';
        this.style.backgroundColor = 'var(--background)';

        const files = e.dataTransfer.files;
        if (files.length > 0) {
            document.getElementById('file-input').files = files;
            handleFileUpload({ target: { files: files } });
        }
    });
}

// ========== Keyboard Shortcuts ==========
document.addEventListener('keydown', function(e) {
    // Alt + Left Arrow: Previous step
    if (e.altKey && e.key === 'ArrowLeft') {
        previousStep();
    }
    // Alt + Right Arrow: Next step
    if (e.altKey && e.key === 'ArrowRight') {
        nextStep();
    }
    // Escape: Close modals
    if (e.key === 'Escape') {
        closeModal();
        closeMissingDataModal();
    }
});

// ========== Utility Functions ==========
function showError(message) {
    alert('Error: ' + message);
}

function showSuccess(message) {
    const alertDiv = document.createElement('div');
    alertDiv.className = 'alert alert-success';
    alertDiv.textContent = message;

    const mainContent = document.getElementById('main-content');
    mainContent.insertBefore(alertDiv, mainContent.firstChild);

    setTimeout(() => {
        alertDiv.remove();
    }, 5000);
}

// Export functions for Python to call
window.appFunctions = {
    populateVariableSelections,
    showMissingDataModal,
    updateProgress,
    showProgress,
    hideProgress,
    showError,
    showSuccess,
    goToStep
};

console.log('JavaScript initialized successfully');
