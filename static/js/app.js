/**
 * OptiGemma — Frontend Application Logic
 * Handles upload, analysis flow, results rendering, and animations.
 */

// =========================================================================
// State
// =========================================================================
let selectedFile = null;
let currentReport = null;
let analysisResult = null;

// =========================================================================
// Upload / Drag & Drop
// =========================================================================
const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const previewArea = document.getElementById('preview-area');
const patientForm = document.getElementById('patient-form');

// Drag & Drop events
dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('drag-over');
});

dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('drag-over');
});

dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('drag-over');
    const files = e.dataTransfer.files;
    if (files.length > 0) handleFile(files[0]);
});

dropZone.addEventListener('click', () => {
    fileInput.click();
});

fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) handleFile(e.target.files[0]);
});

function handleFile(file) {
    if (!file.type.startsWith('image/')) {
        alert('Please upload an image file.');
        return;
    }

    selectedFile = file;

    // Show preview
    const reader = new FileReader();
    reader.onload = (e) => {
        document.getElementById('preview-image').src = e.target.result;
        document.getElementById('preview-name').textContent = file.name;
        previewArea.style.display = 'block';
        dropZone.style.display = 'none';
        patientForm.style.display = 'block';
    };
    reader.readAsDataURL(file);
}

function resetUpload() {
    selectedFile = null;
    fileInput.value = '';
    previewArea.style.display = 'none';
    patientForm.style.display = 'none';
    dropZone.style.display = 'block';
}

// =========================================================================
// Analysis
// =========================================================================
async function analyzeImage() {
    if (!selectedFile) {
        alert('Please select an image first.');
        return;
    }

    // Switch to loading view
    document.getElementById('upload-section').style.display = 'none';
    document.getElementById('results-section').style.display = 'none';
    document.getElementById('loading-section').style.display = 'block';

    // Animate loading steps
    animateLoadingSteps();

    // Build form data
    const formData = new FormData();
    formData.append('image', selectedFile);
    formData.append('age', document.getElementById('patient-age').value || '');
    formData.append('diabetes_duration', document.getElementById('diabetes-duration').value || '');
    formData.append('sugar_level', document.getElementById('sugar-level').value || '');
    formData.append('hba1c', document.getElementById('hba1c').value || '');

    try {
        const response = await fetch('/analyze', {
            method: 'POST',
            body: formData,
        });

        const data = await response.json();

        if (data.error) {
            throw new Error(data.error);
        }

        analysisResult = data;
        currentReport = data.report;

        // Show results
        setTimeout(() => {
            document.getElementById('loading-section').style.display = 'none';
            document.getElementById('results-section').style.display = 'block';
            renderResults(data);
        }, 500);

    } catch (err) {
        console.error('Analysis failed:', err);
        alert('Analysis failed: ' + err.message);
        document.getElementById('loading-section').style.display = 'none';
        document.getElementById('upload-section').style.display = 'block';
    }
}

function animateLoadingSteps() {
    const steps = ['step-preprocess', 'step-detect', 'step-segment', 'step-heatmap', 'step-gemma'];
    let current = 0;

    function activateNext() {
        if (current > 0) {
            document.getElementById(steps[current - 1]).classList.remove('active');
            document.getElementById(steps[current - 1]).classList.add('done');
        }
        if (current < steps.length) {
            document.getElementById(steps[current]).classList.add('active');
            current++;
            setTimeout(activateNext, 800 + Math.random() * 600);
        }
    }

    // Reset all steps
    steps.forEach(id => {
        const el = document.getElementById(id);
        el.classList.remove('active', 'done');
    });

    activateNext();
}

// =========================================================================
// Render Results
// =========================================================================
function renderResults(data) {
    const detection = data.detection;
    const report = data.report;
    const images = data.images;

    // --- Severity Gauge ---
    const stageColors = ['#22c55e', '#eab308', '#f97316', '#ef4444', '#dc2626'];
    const stage = detection.stage;
    const gaugeCircle = document.getElementById('gauge-circle');
    const circumference = 2 * Math.PI * 52; // r=52
    const progress = ((stage + 1) / 5) * circumference;
    const offset = circumference - progress;

    gaugeCircle.style.strokeDasharray = circumference;
    setTimeout(() => {
        gaugeCircle.style.strokeDashoffset = offset;
        gaugeCircle.style.stroke = stageColors[stage] || '#06b6d4';
    }, 100);

    document.getElementById('gauge-stage').textContent = stage;
    document.getElementById('gauge-stage').style.color = stageColors[stage];
    document.getElementById('severity-name').textContent = detection.stage_name;
    document.getElementById('severity-confidence').textContent = `Confidence: ${detection.confidence}%`;

    // Urgency badge
    const urgency = report?.urgency || 'ROUTINE';
    const urgencyBadge = document.getElementById('urgency-badge');
    urgencyBadge.textContent = urgency;
    urgencyBadge.className = `urgency-badge urgency-${urgency}`;

    // Processing time
    document.getElementById('processing-time').textContent = data.processing_time;

    // --- Images ---
    document.getElementById('result-original').src = images.original;
    document.getElementById('result-vessels').src = images.vessels;
    document.getElementById('result-heatmap').src = images.heatmap;

    // Image captions
    if (data.vessel_stats) {
        document.getElementById('vessel-caption').textContent =
            `Vessel density: ${data.vessel_stats.vessel_density_percent}% — ${data.vessel_stats.vessel_health_text}`;
    }
    if (data.heatmap_analysis) {
        document.getElementById('heatmap-caption').textContent =
            `${data.heatmap_analysis.activity_intensity} activity in ${data.heatmap_analysis.most_affected_region}`;
    }

    // --- Probability Bars ---
    renderProbBars(detection.all_probabilities, stage);

    // --- Gemma Report ---
    renderReport(report);
}

function renderProbBars(probs, activeStage) {
    const container = document.getElementById('prob-bars');
    container.innerHTML = '';

    const stageNames = ['No DR', 'Mild NPDR', 'Moderate NPDR', 'Severe NPDR', 'Proliferative DR'];
    const stageColors = ['#22c55e', '#eab308', '#f97316', '#ef4444', '#dc2626'];

    for (let i = 0; i < 5; i++) {
        const prob = probs[i] || 0;
        const row = document.createElement('div');
        row.className = 'prob-bar-row';
        row.innerHTML = `
            <span class="prob-label">${stageNames[i]}</span>
            <div class="prob-bar-track">
                <div class="prob-bar-fill" 
                     style="width: 0%; background: ${stageColors[i]}${i === activeStage ? '' : '80'};"
                     data-value="${prob.toFixed(1)}%">
                </div>
            </div>
        `;
        container.appendChild(row);

        // Animate bar width
        setTimeout(() => {
            row.querySelector('.prob-bar-fill').style.width = `${Math.max(prob, 2)}%`;
        }, 200 + i * 150);
    }
}

function renderReport(report) {
    if (!report) return;

    // Current Diagnosis
    const diagnosis = report.current_diagnosis;
    if (diagnosis) {
        document.getElementById('report-diagnosis').textContent =
            diagnosis.plain_language || `Stage ${diagnosis.stage} — ${diagnosis.stage_name}`;
    }

    // Visual Findings
    const visual = report.visual_findings;
    if (visual) {
        document.getElementById('report-heatmap-findings').textContent =
            visual.heatmap_summary || '—';
        document.getElementById('report-vessel-findings').textContent =
            visual.vessel_analysis || '—';
    }

    // Risk Predictions
    const risk = report.risk_prediction;
    if (risk) {
        const r6 = risk['6_month'] || {};
        document.getElementById('risk-6m-percent').textContent =
            r6.progression_risk_percent || '—';
        document.getElementById('risk-6m-untreated').textContent =
            `⚠️ If untreated: ${r6.scenario_if_untreated || '—'}`;
        document.getElementById('risk-6m-managed').textContent =
            `✅ If managed: ${r6.scenario_if_managed || '—'}`;

        const r12 = risk['12_month'] || {};
        document.getElementById('risk-12m-percent').textContent =
            r12.progression_risk_percent || '—';
        document.getElementById('risk-12m-untreated').textContent =
            `⚠️ If untreated: ${r12.scenario_if_untreated || '—'}`;
        document.getElementById('risk-12m-managed').textContent =
            `✅ If managed: ${r12.scenario_if_managed || '—'}`;
    }

    // Action Plan
    const actionList = document.getElementById('report-actions');
    actionList.innerHTML = '';
    if (report.action_plan && Array.isArray(report.action_plan)) {
        report.action_plan.forEach(item => {
            const li = document.createElement('li');
            li.textContent = item;
            actionList.appendChild(li);
        });
    }

    // Diet
    const dietList = document.getElementById('report-diet');
    dietList.innerHTML = '';
    if (report.diet_recommendations && Array.isArray(report.diet_recommendations)) {
        report.diet_recommendations.forEach(item => {
            const li = document.createElement('li');
            li.textContent = item;
            dietList.appendChild(li);
        });
    }

    // Follow-up
    document.getElementById('report-followup').textContent =
        report.recommended_follow_up || '—';

    // Disclaimer
    if (report.disclaimer) {
        document.getElementById('report-disclaimer').textContent =
            '⚠️ ' + report.disclaimer;
    }
}

// =========================================================================
// Translation
// =========================================================================
async function translateReport() {
    const language = document.getElementById('language-select').value;

    if (language === 'english') {
        renderReport(analysisResult.report);
        return;
    }

    if (!currentReport) return;

    try {
        const response = await fetch('/translate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                report: analysisResult.report,
                language: language,
            }),
        });

        const data = await response.json();
        if (data.success) {
            renderReport(data.report);
        }
    } catch (err) {
        console.error('Translation failed:', err);
    }
}

// =========================================================================
// Reset
// =========================================================================
function resetAll() {
    selectedFile = null;
    currentReport = null;
    analysisResult = null;
    fileInput.value = '';

    // Reset form
    document.getElementById('patient-age').value = '';
    document.getElementById('diabetes-duration').value = '';
    document.getElementById('sugar-level').value = '';
    document.getElementById('hba1c').value = '';
    document.getElementById('language-select').value = 'english';

    // Reset gauge
    document.getElementById('gauge-circle').style.strokeDashoffset = 327;

    // Show upload, hide results
    document.getElementById('upload-section').style.display = 'block';
    document.getElementById('results-section').style.display = 'none';
    document.getElementById('loading-section').style.display = 'none';
    previewArea.style.display = 'none';
    patientForm.style.display = 'none';
    dropZone.style.display = 'block';

    // Scroll to top
    window.scrollTo({ top: 0, behavior: 'smooth' });
}
