//
// Arquiteturas de Alto Desempenho 2025/2026
// File: main.js (Multi-threaded & BigInt Fix)
//

const startButton = document.getElementById('start_button');
const stopButton = document.getElementById('stop_button');
const downloadButton = document.getElementById('download_button');
const customTextEl = document.getElementById('custom_text');
const statusEl = document.getElementById('status');
const attemptsEl = document.getElementById('attempts_counter');
const coinsFoundEl = document.getElementById('coins_found');
const speedValueEl = document.getElementById('speed_value');
const threadCountEl = document.getElementById('thread_count'); 

// --- STATE MANAGEMENT ---
let workers = [];          
let workerStats = [];      
const MAX_COINS_DISPLAY = 50; 

// --- VAULT STORAGE ---
let vaultData = []; 

function escapeHtml(text) {
    if (!text) return text;
    return text
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}

function downloadVault() {
    if (vaultData.length === 0) {
        alert("No coins to download yet!");
        return;
    }
    const fileContent = vaultData.join('\n'); 
    const blob = new Blob([fileContent], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'deti_coins_vault.txt';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

// --- CORREÇÃO AQUI (BigInt) ---
function updateGlobalStats() {
    let totalSpeed = 0;
    let totalAttempts = 0n; // Inicializa como BigInt (notar o 'n')

    for (let stat of workerStats) {
        totalSpeed += stat.speed;
        // Garante que somamos BigInt com BigInt
        totalAttempts += BigInt(stat.attempts);
    }

    const mhs = (totalSpeed / 1000000).toFixed(2);
    // toLocaleString funciona bem com BigInt em browsers modernos
    const att = totalAttempts.toLocaleString('en-US');

    if (speedValueEl) speedValueEl.textContent = mhs;
    else statusEl.textContent = `${mhs} MH/s`;
    
    if (attemptsEl) attemptsEl.textContent = `Attempts: ${att}`;
}

// --- START MINING ---
startButton.onclick = () => {
    if (workers.length > 0) return; 

    const numThreads = threadCountEl ? parseInt(threadCountEl.value) : 1;
    
    // Inicializa stats (attempts com 0n)
    workerStats = new Array(numThreads).fill(null).map(() => ({ speed: 0, attempts: 0n }));

    console.log(`[System] Spawning ${numThreads} workers...`);

    for (let i = 0; i < numThreads; i++) {
        const w = new Worker('worker.js', { type: 'module' });
        w.workerIndex = i;

        w.onerror = function(error) {
            console.error(`WORKER ${i} ERROR:`, error);
        };

        w.onmessage = (e) => {
            const msg = e.data;
            const idx = w.workerIndex;

            if (msg.type === 'status') {
                workerStats[idx].speed = msg.hashesPerSecond; // Number
                workerStats[idx].attempts = msg.totalAttempts; // BigInt (vem do WASM)
                updateGlobalStats();
            } 
            else if (msg.type === 'found') {
                const valStr = msg.value.toString().padStart(2, '0');
                const vaultLine = `V${valStr}:${msg.coin}`;
                vaultData.push(vaultLine);

                downloadButton.disabled = false;

                const safeCoinText = escapeHtml(msg.coin);
                const coinEl = document.createElement('div');
                coinEl.innerHTML = `<b>Value ${msg.value}</b> <span style="font-size:0.8em; color:gray">(T${idx})</span>: ${safeCoinText}<br><small>${msg.hash}</small>`;
                
                coinsFoundEl.prepend(coinEl);

                if (coinsFoundEl.childElementCount > MAX_COINS_DISPLAY) {
                    coinsFoundEl.removeChild(coinsFoundEl.lastChild);
                }
            }
        };

        w.postMessage({
            type: 'start',
            custom_text: customTextEl.value || "",
            worker_id: i
        });

        workers.push(w);
    }

    startButton.disabled = true;
    stopButton.disabled = false;
    
    if (speedValueEl) speedValueEl.textContent = "Starting...";
    if (attemptsEl) attemptsEl.textContent = "Attempts: 0";
};

// --- STOP MINING ---
stopButton.onclick = () => {
    if (workers.length === 0) return;

    console.log("[System] Stopping all workers...");
    
    workers.forEach(w => {
        w.postMessage({ type: 'stop' });
        w.terminate();
    });
    
    workers = [];
    workerStats = [];

    startButton.disabled = false;
    stopButton.disabled = true;
    
    if (speedValueEl) speedValueEl.textContent = "Stopped";
    else if (statusEl) statusEl.textContent = "Stopped.";
};

downloadButton.onclick = downloadVault;