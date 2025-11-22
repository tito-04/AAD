//
// File: main.js (With Vault Download Feature)
//

const startButton = document.getElementById('start_button');
const stopButton = document.getElementById('stop_button');
const downloadButton = document.getElementById('download_button'); // NEW
const customTextEl = document.getElementById('custom_text');
const statusEl = document.getElementById('status');
const attemptsEl = document.getElementById('attempts_counter');
const coinsFoundEl = document.getElementById('coins_found');
const speedValueEl = document.getElementById('speed_value');

let worker = null;
const MAX_COINS_DISPLAY = 50; 

// --- VAULT STORAGE ---
let vaultData = []; // Stores all found coins for download

function escapeHtml(text) {
    if (!text) return text;
    return text
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}

// --- DOWNLOAD FUNCTION ---
function downloadVault() {
    if (vaultData.length === 0) {
        alert("No coins to download yet!");
        return;
    }

    // Create a Blob (File-like object) from the array
    // We join with empty string because the coins usually have their own structure
    // But to ensure clean lines in text editors, usually:
    const fileContent = vaultData.join('\n'); 
    
    const blob = new Blob([fileContent], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    
    // Create invisible link and click it
    const a = document.createElement('a');
    a.href = url;
    a.download = 'deti_coins_vault.txt';
    document.body.appendChild(a);
    a.click();
    
    // Cleanup
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

startButton.onclick = () => {
    if (worker) return; 
    worker = new Worker('worker.js', { type: 'module' }); 
    
    worker.onerror = function(error) {
        console.error("WORKER ERROR:", error);
        alert("Miner stopped due to an error: " + error.message);
        stopButton.click();
    };

    worker.onmessage = (e) => {
        const msg = e.data;
        
        if (msg.type === 'status') {
            const mhs = (msg.hashesPerSecond / 1000000).toFixed(2);
            const attempts = msg.totalAttempts.toLocaleString('en-US'); 
            
            if (speedValueEl) speedValueEl.textContent = mhs;
            else statusEl.textContent = `${mhs} MH/s`;
            
            if (attemptsEl) attemptsEl.textContent = `Attempts: ${attempts}`;
        } 
        else if (msg.type === 'found') {
            // 1. Add to Vault Memory
            // Format: Vxx:COIN_DATA
            const valStr = msg.value.toString().padStart(2, '0');
            // Note: msg.coin might contain the raw non-printable char at the end.
            // We reconstruct the line exactly as the C vault would.
            const vaultLine = `V${valStr}:${msg.coin}`;
            vaultData.push(vaultLine);

            // Enable Download Button
            downloadButton.disabled = false;

            // 2. Update UI
            const safeCoinText = escapeHtml(msg.coin);
            const coinEl = document.createElement('div');
            coinEl.innerHTML = `<b>Value ${msg.value}</b>: ${safeCoinText}<br><small>${msg.hash}</small>`;
            
            coinsFoundEl.prepend(coinEl);

            if (coinsFoundEl.childElementCount > MAX_COINS_DISPLAY) {
                coinsFoundEl.removeChild(coinsFoundEl.lastChild);
            }
        }
        else if (msg.type === 'error') {
            console.error("Worker Error Data:", msg.data);
        }
    };

    worker.postMessage({
        type: 'start',
        custom_text: customTextEl.value || ""
    });

    startButton.disabled = true;
    stopButton.disabled = false;
    
    if (speedValueEl) speedValueEl.textContent = "Starting...";
    if (attemptsEl) attemptsEl.textContent = "Attempts: 0";
};

stopButton.onclick = () => {
    if (worker) {
        worker.postMessage({ type: 'stop' });
        worker.terminate();
        worker = null;
    }

    startButton.disabled = false;
    stopButton.disabled = true;
    
    if (speedValueEl) speedValueEl.textContent = "Stopped";
    else if (statusEl) statusEl.textContent = "Stopped.";
};

// Attach Download Action
downloadButton.onclick = downloadVault;