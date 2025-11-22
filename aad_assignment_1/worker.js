//
// Ficheiro: worker.js (Universal - Auto-detecta funções de string)
//

let wasmModule = null;
let keepRunning = false;
let customText = "";

// Ponteiros e Vistas
let coinBufPtr, hashBufPtr, lcgStatePtr;
let coinView = null;
let hashView = null;
let lcgStateView = null; 

// Estado RNG
let lcgSaltState = null; 
let lcgNonceState = null; 
let totalAccumulatedAttempts = 0n; 

const LCG_MULT = 6364136223846793005n;
const LCG_INCR = 1442695040888963407n;
const SALT_UPDATE_INTERVAL = 10000000000n; 
let saltCounter = 0n; 

function lcg_rand_js(state) {
    return LCG_MULT * state + LCG_INCR;
}

function updateViews() {
    if (!wasmModule) return;
    // Se o buffer foi redimensionado (detached), recriamos as vistas
    if (!coinView || coinView.buffer.byteLength === 0) {
        console.log("DEBUG: Atualizando vistas de memória...");
        coinView = new Uint32Array(wasmModule.HEAPU32.buffer, coinBufPtr, 14);
        hashView = new Uint32Array(wasmModule.HEAPU32.buffer, hashBufPtr, 5);
        lcgStateView = new BigInt64Array(wasmModule.HEAPU32.buffer, lcgStatePtr, 1);
    }
}

async function initWasm() {
    if (wasmModule) return;
    
    console.log("DEBUG: A carregar WebAssembly...");
    const module = await import('./miner.wasm.js');
    wasmModule = await module.default(); 
    console.log("DEBUG: Wasm carregado com sucesso.");

    coinBufPtr = wasmModule._get_coin_buffer_ptr();
    hashBufPtr = wasmModule._get_hash_buffer_ptr();
    lcgStatePtr = wasmModule._get_lcg_state_ptr(); 
}

function getCoinString() {
    updateViews(); 
    let s = "";
    const coinBytes = new Uint8Array(wasmModule.HEAPU8.buffer, coinBufPtr, 14 * 4);
    for (let i = 0; i < 55; i++) {
        s += String.fromCharCode(coinBytes[i ^ 3]);
    }
    return s.replace('\n', '');
}

async function startMining() {
    try {
        await initWasm();
    } catch (e) {
        console.error("CRITICAL ERROR: Falha ao carregar Wasm:", e);
        postMessage({ type: 'error', data: "Wasm Load Failed: " + e.toString() });
        return;
    }

    // --- SOLUÇÃO UNIVERSAL PARA STRINGS ---
    let customTextPtr;
    try {
        if (typeof wasmModule.stringToNewUTF8 === 'function') {
            console.log("DEBUG: Usando stringToNewUTF8");
            customTextPtr = wasmModule.stringToNewUTF8(customText);
        } else if (typeof wasmModule.allocateUTF8 === 'function') {
            console.log("DEBUG: Usando allocateUTF8");
            customTextPtr = wasmModule.allocateUTF8(customText);
        } else {
            throw new Error("Nenhuma função de alocação de string encontrada no Wasm!");
        }
    } catch (err) {
        console.error("Erro ao alocar string:", err);
        postMessage({ type: 'error', data: "String Alloc Failed: " + err.message });
        return;
    }

    // Atualizar vistas APÓS alocação (pois a memória pode ter crescido)
    updateViews();

    const customLen = customText.length > 34 ? 34 : customText.length;
    const chunkSize = 500000; 
    keepRunning = true;

    if (lcgNonceState === null) {
        lcgSaltState = BigInt(Math.floor(Math.random() * 2**32)) * 2n**32n + BigInt(Date.now());
        lcgNonceState = lcgSaltState; 
    }

    console.log("DEBUG: Loop de mineração iniciado.");
    let totalHashes = 0; 
    let startTime = performance.now();

    while (keepRunning) {
        if (saltCounter >= SALT_UPDATE_INTERVAL) {
            for(let i=0; i<34; i++) lcgSaltState = lcg_rand_js(lcgSaltState);
            saltCounter = 0n;
        }

        const nLow = Number(lcgNonceState & 0xFFFFFFFFn);
        const nHigh = Number(lcgNonceState >> 32n);
        const sLow = Number(lcgSaltState & 0xFFFFFFFFn);
        const sHigh = Number(lcgSaltState >> 32n);

        // Executa C
        const value = wasmModule._search_chunk(nLow, nHigh, sLow, sHigh, customTextPtr, customLen, chunkSize);
        
        // Atualiza estado
        updateViews(); 
        lcgNonceState = lcgStateView[0]; 

        if (value >= 0) { 
            const coinString = getCoinString();
            const hashHex = Array.from(hashView).map(h => h.toString(16).padStart(8, '0')).join('');
            postMessage({ type: 'found', value: value, hash: hashHex, coin: coinString });
            lcgNonceState = lcg_rand_js(lcgNonceState);
        }

        totalHashes += chunkSize;
        saltCounter += BigInt(chunkSize);
        totalAccumulatedAttempts += BigInt(chunkSize); 

        const timeNow = performance.now();
        if (timeNow - startTime > 1000) {
            const hashesPerSecond = totalHashes / ((timeNow - startTime) / 1000);
            // Envia status para atualizar o "Starting..."
            postMessage({ type: 'status', hashesPerSecond: hashesPerSecond, totalAttempts: totalAccumulatedAttempts });
            totalHashes = 0;
            startTime = timeNow;
        }
    }
    
    wasmModule._free(customTextPtr);
}

self.onmessage = (e) => {
    const msg = e.data;
    if (msg.type === 'start') {
        if (!keepRunning) {
            customText = msg.custom_text || "";
            startMining();
        }
    } else if (msg.type === 'stop') {
        keepRunning = false;
    }
};