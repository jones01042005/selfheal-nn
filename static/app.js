/**
 * SelfHealingNN — Frontend App Logic v2.0
 * ----------------------------------------
 * Communicates with the FastAPI backend (api.py).
 * All API calls go to /api/* (works both locally and when hosted).
 */

// ── Configuration ────────────────────────────────────────────────
const API_BASE = (window.BACKEND_URL || '').replace(/\/$/, '');

// Method metadata (mirrors /api/methods response)
const METHOD_META = {
    adaptive:          { icon: '🤖', name: 'Adaptive Controller',     ae: false, hint: 'Automatically picks the optimal strategy based on fault severity.' },
    gradient:          { icon: '📉', name: 'Gradient-Based Repair',   ae: false, hint: 'Reinitialises low-gradient faulty weights with small random values.' },
    regrowth:          { icon: '🌱', name: 'Neuron Regrowth',         ae: false, hint: 'Reinitialises neurons where >90% of incoming weights are dead.' },
    regrowth_gradient: { icon: '🌱', name: 'Regrowth + Gradient',    ae: false, hint: 'Neuron regrowth followed by targeted gradient fine-repair.' },
    rerouting:         { icon: '🔀', name: 'Dynamic Rerouting',       ae: false, hint: 'Scales down damaged neuron outputs to redirect information flow.' },
    ae_dual:           { icon: '🔁', name: 'Dual AE Reconstruction',  ae: true,  hint: 'Uses pre-trained autoencoders to reconstruct the FC1 and FC2 weight matrices.' },
    hybrid:            { icon: '⚡', name: 'Hybrid (AE + Gradient)',  ae: true,  hint: 'AE reconstruction followed by gradient fine-repair — highest accuracy.' },
};

const FAULT_MAP = { 1: 10, 2: 20, 3: 30, 4: 40 };

// ── DOM References ────────────────────────────────────────────────
const $ = id => document.getElementById(id);

const dropzone    = $('dropzone');
const fileInput   = $('image-upload');
const imgPreview  = $('image-preview');
const dzInner     = $('dropzone-inner');

const faultSlider = $('fault-ratio');
const faultBadge  = $('fault-badge');
const healMethod  = $('heal-method');
const methodHint  = $('method-hint');
const methodInfo  = $('method-info');
const miIcon      = $('mi-icon');
const miTitle     = $('mi-title');
const miDesc      = $('mi-desc');
const miAeBadge   = $('mi-ae-badge');

const btnBaseline = $('btn-baseline');
const btnFault    = $('btn-fault');
const btnHeal     = $('btn-heal');
const statusBar   = $('status-bar');
const statusText  = $('status-text');
const errorBar    = $('error-bar');
const errorText   = $('error-text');

const sectionResults = $('section-results');
const cardBaseline   = $('card-baseline');
const cardFault      = $('card-fault');
const cardHealed     = $('card-healed');

const baselineAcc = $('baseline-acc');
const baselinePred= $('baseline-pred');
const faultAcc    = $('fault-acc');
const faultPred   = $('fault-pred');
const healedAcc   = $('healed-acc');
const healedPred  = $('healed-pred');
const recoveryRow = $('recovery-row');
const recoveryVal = $('recovery-val');

const summaryRow  = $('summary-row');
const sumMethod   = $('sum-method');
const sumFault    = $('sum-fault');
const sumDrop     = $('sum-drop');
const sumGain     = $('sum-gain');
const sumTime     = $('sum-time');

const chartSection = $('chart-section');
let resultsChart   = null;

// Cached results across steps
let _baselineAccVal = null;
let _faultAccVal    = null;

// ── Image Upload ─────────────────────────────────────────────────
let selectedFile = null;

function handleFile(file) {
    if (!file || !file.type.startsWith('image/')) return;
    selectedFile = file;

    const reader = new FileReader();
    reader.onload = e => {
        imgPreview.src = e.target.result;
        imgPreview.classList.remove('hidden');
        dzInner.classList.add('hidden');

        // Unlock step 1
        btnBaseline.disabled = false;
        btnFault.disabled    = true;
        btnHeal.disabled     = true;

        // Reset results
        resetResults();
        hideStatus();
    };
    reader.readAsDataURL(file);
}

dropzone.addEventListener('click', e => {
    if (e.target !== fileInput) fileInput.click();
});
dropzone.addEventListener('keydown', e => {
    if (e.key === 'Enter' || e.key === ' ') fileInput.click();
});
dropzone.addEventListener('dragover', e => {
    e.preventDefault();
    dropzone.style.borderColor = 'var(--c-blue)';
});
dropzone.addEventListener('dragleave', e => {
    dropzone.style.borderColor = '';
});
dropzone.addEventListener('drop', e => {
    e.preventDefault();
    dropzone.style.borderColor = '';
    if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
});
fileInput.addEventListener('change', e => {
    if (e.target.files.length) handleFile(e.target.files[0]);
});

// ── Fault Slider ─────────────────────────────────────────────────
faultSlider.addEventListener('input', () => {
    const pct = FAULT_MAP[faultSlider.value];
    faultBadge.textContent = pct + '%';
});

// ── Method Selector ───────────────────────────────────────────────
function updateMethodUI() {
    const m = healMethod.value;
    const meta = METHOD_META[m] || {};
    methodHint.textContent = meta.hint || '';

    miIcon.textContent  = meta.icon  || '?';
    miTitle.textContent = meta.name  || m;
    miDesc.textContent  = meta.hint  || '';

    if (meta.ae) {
        miAeBadge.style.display = '';
    } else {
        miAeBadge.style.display = 'none';
    }
    methodInfo.classList.remove('hidden');
}
healMethod.addEventListener('change', updateMethodUI);
updateMethodUI(); // init

// ── Chart Initialisation ──────────────────────────────────────────
function initChart() {
    if (resultsChart) {
        resultsChart.destroy();
        resultsChart = null;
    }
    const ctx = $('results-chart').getContext('2d');
    resultsChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['Baseline', 'Faulty', 'Healed'],
            datasets: [{
                label: 'Accuracy (%)',
                data: [null, null, null],
                backgroundColor: [
                    'rgba(59,130,246,.75)',
                    'rgba(239,68,68,.75)',
                    'rgba(16,185,129,.75)',
                ],
                borderColor: [
                    '#3b82f6',
                    '#ef4444',
                    '#10b981',
                ],
                borderWidth: 2,
                borderRadius: 8,
                borderSkipped: false,
            }]
        },
        options: {
            responsive: true,
            animation: { duration: 800, easing: 'easeOutQuart' },
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        label: ctx => ` Accuracy: ${ctx.parsed.y.toFixed(2)}%`
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100,
                    grid: { color: 'rgba(255,255,255,.06)' },
                    ticks: {
                        color: '#7a8aa8',
                        callback: v => v + '%'
                    }
                },
                x: {
                    grid: { display: false },
                    ticks: {
                        color: '#e8edf8',
                        font: { size: 13, weight: '600' }
                    }
                }
            }
        }
    });
}

function setChartBar(index, value) {
    if (!resultsChart) return;
    resultsChart.data.datasets[0].data[index] = value;
    resultsChart.update();
}

// ── UI Helpers ────────────────────────────────────────────────────
function showStatus(msg) {
    statusText.textContent = msg;
    statusBar.classList.remove('hidden');
    errorBar.classList.add('hidden');
}
function hideStatus() {
    statusBar.classList.add('hidden');
}
function showError(msg) {
    errorText.textContent = msg;
    errorBar.classList.remove('hidden');
    statusBar.classList.add('hidden');
}

function activateCard(card) {
    card.classList.remove('pending');
    card.classList.add('active');
    const stateEl = card.querySelector('.card-state');
    if (stateEl) {
        stateEl.textContent = '✓ Complete';
        stateEl.classList.remove('pending-state');
        stateEl.classList.add('done-state');
    }
}

function resetResults() {
    [cardBaseline, cardFault, cardHealed].forEach(c => {
        c.classList.remove('active');
        c.classList.add('pending');
        const stateEl = c.querySelector('.card-state');
        if (stateEl) {
            stateEl.textContent = 'Waiting…';
            stateEl.classList.add('pending-state');
            stateEl.classList.remove('done-state');
        }
    });
    baselineAcc.textContent = '—';
    baselinePred.textContent = '—';
    faultAcc.textContent = '—';
    faultPred.textContent = '—';
    healedAcc.textContent = '—';
    healedPred.textContent = '—';
    recoveryRow.classList.add('hidden');
    summaryRow.classList.add('hidden');
    chartSection.classList.add('hidden');
    sectionResults.classList.remove('hidden');
    _baselineAccVal = null;
    _faultAccVal    = null;
}

function getFaultPct() { return FAULT_MAP[faultSlider.value]; }
function getMethod()    { return healMethod.value; }

// ── API helpers ───────────────────────────────────────────────────
function buildFormData(extras = {}) {
    const fd = new FormData();
    fd.append('image', selectedFile);
    Object.entries(extras).forEach(([k, v]) => fd.append(k, v));
    return fd;
}

async function apiPost(path, formData) {
    const url = API_BASE + path;
    const res = await fetch(url, { method: 'POST', body: formData });

    let responseBody;
    try {
        responseBody = await res.text();
    } catch {
        throw new Error(`Unable to read server response (HTTP ${res.status}).`);
    }

    let data = null;
    if (responseBody.trim()) {
        try {
            data = JSON.parse(responseBody);
        } catch {
            const bodyPreview = responseBody.trim().slice(0, 240);
            if (!res.ok) {
                throw new Error(`HTTP ${res.status}: ${bodyPreview}`);
            }
            throw new Error(`Invalid JSON response (HTTP ${res.status}).`);
        }
    }

    if (!res.ok) {
        const detail = data && (data.detail || data.error);
        const message = typeof detail === 'string'
            ? detail
            : detail
                ? JSON.stringify(detail)
                : responseBody.trim()
                    ? responseBody.trim().slice(0, 240)
                    : 'The server returned an empty response.';
        throw new Error(`HTTP ${res.status}: ${message}`);
    }

    if (!data || typeof data !== 'object' || Array.isArray(data)) {
        throw new Error(`Invalid or empty server response (HTTP ${res.status}).`);
    }
    if (data.error) {
        throw new Error(String(data.error));
    }

    return data;
}

// ── Step 1: Baseline ──────────────────────────────────────────────
btnBaseline.addEventListener('click', async () => {
    if (!selectedFile) return;
    btnBaseline.disabled = true;
    btnFault.disabled    = true;
    btnHeal.disabled     = true;

    resetResults();
    showStatus('Evaluating baseline CNN on CIFAR-10 validation set…');
    initChart();
    chartSection.classList.remove('hidden');

    try {
        const data = await apiPost('/api/baseline', buildFormData());
        _baselineAccVal = data.accuracy;

        baselineAcc.textContent  = data.accuracy + '%';
        baselinePred.textContent = data.prediction;
        activateCard(cardBaseline);
        setChartBar(0, data.accuracy);

        btnFault.disabled = false;
        hideStatus();
    } catch (e) {
        showError('Baseline failed: ' + e.message);
        btnBaseline.disabled = false;
    }
});

// ── Step 2: Fault Injection ────────────────────────────────────────
btnFault.addEventListener('click', async () => {
    if (!selectedFile) return;
    btnFault.disabled = true;
    btnHeal.disabled  = true;

    const pct = getFaultPct();
    showStatus(`Injecting ${pct}% weight fault into a fresh model copy…`);

    try {
        const data = await apiPost('/api/fault', buildFormData({ fault_ratio: pct }));
        _faultAccVal = data.accuracy;

        faultAcc.textContent  = data.accuracy + '%';
        faultPred.textContent = data.prediction;
        activateCard(cardFault);
        setChartBar(1, data.accuracy);

        btnHeal.disabled = false;
        hideStatus();
    } catch (e) {
        showError('Fault injection failed: ' + e.message);
        btnFault.disabled = false;
    }
});

// ── Step 3: Heal ──────────────────────────────────────────────────
btnHeal.addEventListener('click', async () => {
    if (!selectedFile) return;
    btnHeal.disabled = true;

    const pct    = getFaultPct();
    const method = getMethod();
    const meta   = METHOD_META[method] || {};

    showStatus(`Running "${meta.name || method}" healing strategy (fault=${pct}%)… this may take 20–60s`);

    try {
        const data = await apiPost('/api/heal', buildFormData({ fault_ratio: pct, method }));

        healedAcc.textContent  = data.healed_acc + '%';
        healedPred.textContent = data.prediction;
        recoveryVal.textContent = data.recovery + '%';
        recoveryRow.classList.remove('hidden');
        activateCard(cardHealed);
        setChartBar(2, data.healed_acc);

        // Update chart with real baseline & faulty values from this response
        if (data.baseline_acc !== undefined) setChartBar(0, data.baseline_acc);
        if (data.faulty_acc   !== undefined) setChartBar(1, data.faulty_acc);

        // Summary row
        const drop = ((data.baseline_acc || _baselineAccVal || 0) - data.faulty_acc).toFixed(2);
        const gain = (data.healed_acc - data.faulty_acc).toFixed(2);
        sumMethod.textContent = meta.name || method;
        sumFault.textContent  = pct + '%';
        sumDrop.textContent   = '-' + Math.abs(drop) + '%';
        sumGain.textContent   = '+' + Math.abs(gain) + '%';
        sumTime.textContent   = data.heal_time_s !== undefined ? data.heal_time_s + 's' : '—';
        summaryRow.classList.remove('hidden');

        hideStatus();
    } catch (e) {
        showError('Healing failed: ' + e.message);
        btnHeal.disabled = false;
    }
});
