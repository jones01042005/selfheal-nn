const dropzone = document.getElementById('dropzone');
const fileInput = document.getElementById('image-upload');
const imagePreview = document.getElementById('image-preview');
const dropzoneText = dropzone.querySelector('p');

const btn1 = document.getElementById('btn-1');
const btn2 = document.getElementById('btn-2');
const btn3 = document.getElementById('btn-3');

const loadingText = document.getElementById('loading-text');
const resultsPanel = document.getElementById('results-panel');
const plotContainer = document.getElementById('plot-container');

const cardBase = document.getElementById('card-base');
const cardFault = document.getElementById('card-fault');
const cardHeal = document.getElementById('card-heal');

let selectedFile = null;

// Initialize Chart.js Instance
const ctx = document.getElementById('resultsChart').getContext('2d');
let diagnosticsChart = new Chart(ctx, {
    type: 'bar',
    data: {
        labels: ['Baseline Accuracy', 'Hardware Fault', 'Healed Network'],
        datasets: [{
            label: 'Evaluation Accuracy (%)',
            data: [0, 0, 0],
            backgroundColor: [
                'rgba(59, 130, 246, 0.8)', // Blue
                'rgba(239, 68, 68, 0.8)',   // Red
                'rgba(16, 185, 129, 0.8)'   // Green
            ],
            borderColor: [
                'rgba(59, 130, 246, 1)',
                'rgba(239, 68, 68, 1)',
                'rgba(16, 185, 129, 1)'
            ],
            borderWidth: 1,
            borderRadius: 6
        }]
    },
    options: {
        responsive: true,
        plugins: {
            legend: { display: false },
            tooltip: {
                callbacks: {
                    label: function(context) {
                        return ' Accuracy: ' + context.parsed.y + '%';
                    }
                }
            }
        },
        scales: {
            y: {
                beginAtZero: true,
                max: 100,
                grid: {
                    color: 'rgba(255, 255, 255, 0.1)'
                },
                ticks: {
                    color: '#94a3b8'
                }
            },
            x: {
                grid: { display: false },
                ticks: { color: '#f8fafc', font: { size: 14 } }
            }
        }
    }
});

// Handle Drag and Drop
dropzone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropzone.style.borderColor = 'var(--accent)';
});

dropzone.addEventListener('dragleave', (e) => {
    e.preventDefault();
    dropzone.style.borderColor = 'rgba(255, 255, 255, 0.2)';
});

dropzone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropzone.style.borderColor = 'rgba(255, 255, 255, 0.2)';
    
    if (e.dataTransfer.files.length > 0) {
        handleFileSelect(e.dataTransfer.files[0]);
    }
});

dropzone.addEventListener('click', () => {
    fileInput.click();
});

fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        handleFileSelect(e.target.files[0]);
    }
});

function handleFileSelect(file) {
    if (!file.type.startsWith('image/')) return;
    
    selectedFile = file;
    
    const reader = new FileReader();
    reader.onload = (e) => {
        imagePreview.src = e.target.result;
        imagePreview.classList.remove('hidden');
        dropzoneText.classList.add('hidden');
        
        // Reset state
        btn1.disabled = false;
        btn2.disabled = true;
        btn3.disabled = true;
        
        resultsPanel.classList.add('hidden');
        plotContainer.classList.add('hidden');
        
        // Reset Visual Styles
        cardBase.style.opacity = '0.5';
        cardFault.style.opacity = '0.5';
        cardHeal.style.opacity = '0.5';

        // Reset Data Chart
        diagnosticsChart.data.datasets[0].data = [0, 0, 0];
        diagnosticsChart.update();
    };
    reader.readAsDataURL(file);
}

// Global form data constructor
function getFormData() {
    const faultRatio = document.getElementById('fault-ratio').value;
    const formData = new FormData();
    formData.append('image', selectedFile);
    formData.append('fault_ratio', faultRatio);
    return formData;
}

// 1. Classify Baseline
btn1.addEventListener('click', async () => {
    if (!selectedFile) return;
    
    btn1.disabled = true;
    loadingText.textContent = "Running global baseline inference...";
    loadingText.classList.remove('hidden');
    resultsPanel.classList.remove('hidden');
    plotContainer.classList.remove('hidden');
    
    try {
        const res = await fetch('/api/baseline', { method: 'POST', body: getFormData() });
        const data = await res.json();
        if(data.error) throw new Error(data.error);
        
        document.getElementById('base-acc').textContent = data.accuracy + '%';
        document.getElementById('base-pred').textContent = data.prediction;
        cardBase.style.opacity = '1';
        
        // Map Chart
        diagnosticsChart.data.datasets[0].data[0] = parseFloat(data.accuracy);
        diagnosticsChart.update();

        btn2.disabled = false; // unlock next phase
    } catch (e) {
        alert(e);
    } finally {
        loadingText.classList.add('hidden');
    }
});

// 2. Simulate Hardware Fault
btn2.addEventListener('click', async () => {
    btn2.disabled = true;
    loadingText.textContent = "Cloning VRAM block and simulating weights drop...";
    loadingText.classList.remove('hidden');
    
    try {
        const res = await fetch('/api/fault', { method: 'POST', body: getFormData() });
        const data = await res.json();
        if(data.error) throw new Error(data.error);
        
        document.getElementById('fault-acc').textContent = data.accuracy + '%';
        document.getElementById('fault-pred').textContent = data.prediction;
        cardFault.style.opacity = '1';
        
        // Map Chart
        diagnosticsChart.data.datasets[0].data[1] = parseFloat(data.accuracy);
        diagnosticsChart.update();

        btn3.disabled = false; // unlock next phase
    } catch (e) {
        alert(e);
    } finally {
        loadingText.classList.add('hidden');
    }
});

// 3. Self-Heal Network
btn3.addEventListener('click', async () => {
    btn3.disabled = true;
    loadingText.textContent = "Tracing gradients and surgically repairing topology...";
    loadingText.classList.remove('hidden');
    
    try {
        const res = await fetch('/api/heal', { method: 'POST', body: getFormData() });
        const data = await res.json();
        if(data.error) throw new Error(data.error);
        
        document.getElementById('heal-acc').textContent = data.accuracy + '%';
        document.getElementById('heal-pred').textContent = data.prediction;
        document.getElementById('heal-rec').textContent = data.recovery + '%';
        cardHeal.style.opacity = '1';
        
        // Map Chart
        diagnosticsChart.data.datasets[0].data[2] = parseFloat(data.accuracy);
        diagnosticsChart.update();

    } catch (e) {
        alert(e);
    } finally {
        loadingText.classList.add('hidden');
    }
});
