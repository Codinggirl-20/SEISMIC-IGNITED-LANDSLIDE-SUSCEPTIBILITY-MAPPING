/**
 * SILSM - Seismic-Ignited Landslide Susceptibility Mapping Dashboard
 * Core Application Script - Manages GIS State, Earthquake Attenuation,
 * Map Canvas Rendering, and Analytical Charts.
 */

// Global Dashboard State
const state = {
    gridCols: 45,
    gridRows: 30,
    cellSize: 13, // pixels per grid cell
    grid: [],     // Array containing cell data objects
    epicenter: null, // {col, row}
    seismicWaveRadius: 0,
    isSimulating: false,
    activeLayer: 'terrain', // 'terrain', 'slope', 'ndvi', 'sar', 'susceptibility'
    
    // EQ Parameters
    eqMagnitude: 6.8,
    eqDepth: 12,
    soilClass: 'C',
    maxPGA: 0.0,
    
    // UI elements
    hoveredCell: null,
    selectedCell: null
};

// --- Initial Setup and Event Bindings ---
window.addEventListener('DOMContentLoaded', () => {
    initTerrainGrid();
    initUIControls();
    renderLegend();
    
    // Initial draw
    drawMap();
    updateDashboardMetrics();
    
    // Set up canvas mouse event handlers
    const canvas = document.getElementById('map-canvas');
    canvas.addEventListener('mousemove', handleMapHover);
    canvas.addEventListener('click', handleMapClick);
    
    // Initialize model visualization canvas
    if (window.initModelVisualizer) {
        window.initModelVisualizer();
    }
});

// --- Terrain Generation ---
// Generates a mock mountainous grid with spatial coherence (valleys & ridges)
function initTerrainGrid() {
    state.grid = [];
    const cols = state.gridCols;
    const rows = state.gridRows;
    
    // Generate base ridges using sine waves and noise
    for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
            // Normalized positions (0 to 1)
            const nx = c / cols;
            const ny = r / rows;
            
            // Elevation formula simulating a diagonal mountain ridge cutting across the region
            const ridge1 = Math.sin(nx * Math.PI * 1.5 + ny * Math.PI) * 800;
            const ridge2 = Math.cos((nx - 0.5) * Math.PI * 3) * 400;
            const noise = (Math.sin(c * 0.4) * Math.cos(r * 0.4) * 150) + (Math.sin(c * 1.2) * 50);
            
            let elevation = 1200 + ridge1 + ridge2 + noise;
            elevation = Math.max(500, Math.min(2600, elevation)); // clamp between 500m and 2600m
            
            // Lithology distribution: bottom-right has loose colluvium (clay), 
            // ridges have hard granite, valleys have sandstone.
            let lithology = 1; // Sandstone (default)
            if (elevation > 1900) {
                lithology = 0; // Hard Granite
            } else if (r > rows * 0.7 && c > cols * 0.6) {
                lithology = 2; // Loose Colluvium / Clay
            }
            
            // Soil stiffness (NEHRP) distributed logically: lower elevations/valleys have softer soil
            let cellSoil = 'C';
            if (elevation < 900) {
                cellSoil = 'D'; // Soft/stiff soil in river valleys
            } else if (elevation > 2000) {
                cellSoil = 'B'; // Rock
            }
            
            state.grid.push({
                col: c,
                row: r,
                elevation: Math.round(elevation),
                slope: 0,       // Calculated below
                aspect: 0,      // Calculated below
                ndvi: 0,        // Calculated below
                vs30: cellSoil === 'B' ? 820 : (cellSoil === 'C' ? 490 : 250),
                soilClass: cellSoil,
                lithology: lithology,
                
                // Pre-event satellite coherence
                coherencePre: 0.78 + (Math.random() * 0.15),
                coherencePost: 0.78 + (Math.random() * 0.15),
                coherenceLoss: 0.05 + (Math.random() * 0.05), // base decorrelation noise
                
                // Dynamic ground motion values
                pga: 0.0,
                arias: 0.0,
                susceptibility: 0.05 // low base probability
            });
        }
    }
    
    // Calculate slopes & aspect based on neighboring elevations (finite difference method)
    for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
            const idx = r * cols + c;
            const cell = state.grid[idx];
            
            // Get elevations of direct neighbors
            const z_left = c > 0 ? state.grid[r * cols + (c - 1)].elevation : cell.elevation;
            const z_right = c < cols - 1 ? state.grid[r * cols + (c + 1)].elevation : cell.elevation;
            const z_up = r > 0 ? state.grid[(r - 1) * cols + c].elevation : cell.elevation;
            const z_down = r < rows - 1 ? state.grid[(r + 1) * cols + c].elevation : cell.elevation;
            
            // Gradient in x and y (30m spatial resolution simulated)
            const dx = (z_right - z_left) / 60.0;
            const dy = (z_down - z_up) / 60.0;
            
            // Slope in degrees
            const slopeRad = Math.atan(Math.sqrt(dx*dx + dy*dy));
            cell.slope = Math.round(slopeRad * (180.0 / Math.PI));
            
            // Aspect (azimuth angle of the slope direction)
            let aspectDeg = Math.round(Math.atan2(-dy, dx) * (180.0 / Math.PI));
            if (aspectDeg < 0) aspectDeg += 360;
            cell.aspect = aspectDeg;
            
            // NDVI: vegetation density. Higher on flat valleys, lower on steep slopes & high elevations
            let ndvi = 0.75 - (cell.slope / 70.0) * 0.4 - (Math.abs(cell.elevation - 1000) / 1600.0) * 0.3;
            ndvi += (Math.random() * 0.08) - 0.04; // add micro texture
            cell.ndvi = Math.max(0.05, Math.min(0.85, ndvi));
            cell.coherencePre = 0.85 - (cell.ndvi * 0.15) + (Math.random() * 0.05); // veg decreases coherence
            cell.coherencePost = cell.coherencePre - cell.coherenceLoss;
        }
    }
}

// --- UI Binding and Event Listeners ---
function initUIControls() {
    // Inputs & Sliders
    const sliderMag = document.getElementById('eq-magnitude');
    const labelMag = document.getElementById('val-magnitude');
    sliderMag.addEventListener('input', (e) => {
        state.eqMagnitude = parseFloat(e.target.value);
        labelMag.innerText = state.eqMagnitude.toFixed(1);
    });

    const sliderDepth = document.getElementById('eq-depth');
    const labelDepth = document.getElementById('val-depth');
    sliderDepth.addEventListener('input', (e) => {
        state.eqDepth = parseInt(e.target.value);
        labelDepth.innerText = state.eqDepth + " km";
    });
    
    document.getElementById('soil-class').addEventListener('change', (e) => {
        state.soilClass = e.target.value;
    });

    // Layer selection buttons
    const layerButtons = document.querySelectorAll('.layer-btn');
    layerButtons.forEach(btn => {
        btn.addEventListener('click', (e) => {
            layerButtons.forEach(b => b.classList.remove('active'));
            const targetBtn = e.currentTarget;
            targetBtn.classList.add('active');
            state.activeLayer = targetBtn.dataset.layer;
            drawMap();
            renderLegend();
        });
    });

    // Epicenter button
    document.getElementById('btn-random-epicenter').addEventListener('click', () => {
        state.epicenter = {
            col: Math.floor(10 + Math.random() * (state.gridCols - 20)),
            row: Math.floor(8 + Math.random() * (state.gridRows - 16))
        };
        document.getElementById('stat-epicenter').innerText = `Col ${state.epicenter.col}, Row ${state.epicenter.row}`;
        drawMap();
    });

    // Action triggers
    document.getElementById('btn-trigger-eq').addEventListener('click', triggerEarthquake);
    document.getElementById('btn-reset').addEventListener('click', resetSimulation);
}

// --- Map Drawing Logic ---
function drawMap() {
    const canvas = document.getElementById('map-canvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    
    // Clear canvas
    ctx.fillStyle = '#07090e';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    
    const cols = state.gridCols;
    const rows = state.gridRows;
    const size = state.cellSize;
    
    // Offset for centering grid in canvas
    const offsetX = Math.floor((canvas.width - (cols * size)) / 2);
    const offsetY = Math.floor((canvas.height - (rows * size)) / 2);
    
    // Draw cells
    for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
            const idx = r * cols + c;
            const cell = state.grid[idx];
            
            ctx.fillStyle = getCellColor(cell);
            ctx.fillRect(offsetX + c * size, offsetY + r * size, size - 1, size - 1);
            
            // Draw hover border
            if (state.hoveredCell && state.hoveredCell.col === c && state.hoveredCell.row === r) {
                ctx.strokeStyle = '#ffffff';
                ctx.lineWidth = 1.5;
                ctx.strokeRect(offsetX + c * size - 1, offsetY + r * size - 1, size + 1, size + 1);
            }
            // Draw selection border
            if (state.selectedCell && state.selectedCell.col === c && state.selectedCell.row === r) {
                ctx.strokeStyle = '#3b82f6';
                ctx.lineWidth = 2;
                ctx.strokeRect(offsetX + c * size - 1, offsetY + r * size - 1, size + 1, size + 1);
            }
        }
    }
    
    // Draw epicenter symbol if placed
    if (state.epicenter) {
        const epX = offsetX + state.epicenter.col * size + size/2;
        const epY = offsetY + state.epicenter.row * size + size/2;
        
        // Draw expanding wave animation ring
        if (state.isSimulating) {
            ctx.strokeStyle = 'rgba(249, 115, 22, 0.8)';
            ctx.lineWidth = 2.5;
            ctx.beginPath();
            ctx.arc(epX, epY, state.seismicWaveRadius * size, 0, 2 * Math.PI);
            ctx.stroke();
        }
        
        // Epicenter star symbol
        ctx.fillStyle = '#f97316';
        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth = 1.5;
        
        ctx.beginPath();
        drawStar(ctx, epX, epY, 5, 12, 5);
        ctx.fill();
        ctx.stroke();
    }
}

// Helper: Draw a star path for epicenter marker
function drawStar(ctx, cx, cy, spikes, outerRadius, innerRadius) {
    let rot = Math.PI / 2 * 3;
    let x = cx;
    let y = cy;
    let step = Math.PI / spikes;

    ctx.moveTo(cx, cy - outerRadius);
    for (let i = 0; i < spikes; i++) {
        x = cx + Math.cos(rot) * outerRadius;
        y = cy + Math.sin(rot) * outerRadius;
        ctx.lineTo(x, y);
        rot += step;

        x = cx + Math.cos(rot) * innerRadius;
        y = cy + Math.sin(rot) * innerRadius;
        ctx.lineTo(x, y);
        rot += step;
    }
    ctx.lineTo(cx, cy - outerRadius);
    ctx.closePath();
}

// Maps cell attributes to solid color schemes based on the active display layer
function getCellColor(cell) {
    switch (state.activeLayer) {
        case 'terrain': {
            // Shaded Relief representation (hillshading)
            // Simulates sun illumination from northwest (315 azimuth, 45 elevation)
            const sunAzimuthRad = 315 * (Math.PI / 180.0);
            const sunElevationRad = 45 * (Math.PI / 180.0);
            
            const slopeRad = cell.slope * (Math.PI / 180.0);
            const aspectRad = cell.aspect * (Math.PI / 180.0);
            
            // Calculate cosine of solar incidence angle (shading value)
            const cosIncidence = Math.cos(sunElevationRad) * Math.cos(slopeRad) +
                                 Math.sin(sunElevationRad) * Math.sin(slopeRad) * Math.cos(sunAzimuthRad - aspectRad);
                                 
            // Map incidence to a shading factor [0.4 to 1.1]
            const shade = 0.4 + Math.max(0, cosIncidence) * 0.7;
            
            // Base altitude color mapping (valleys are dark green-gray, mountain ridges are light gray)
            const heightPercent = (cell.elevation - 500) / 2100.0;
            
            // Base RGB colors
            let r = 26, g = 38, b = 58; // low altitude: slate blue valley
            if (heightPercent > 0.4) {
                // mid altitude: forest/hills
                r = 45; g = 55; b = 72;
            }
            if (heightPercent > 0.75) {
                // high altitude: bare rock/cliffs
                r = 100; g = 110; b = 120;
            }
            
            // Apply hillshading factor
            const finalR = Math.min(255, Math.round(r * shade));
            const finalG = Math.min(255, Math.round(g * shade));
            const finalB = Math.min(255, Math.round(b * shade));
            
            return `rgb(${finalR}, ${finalG}, ${finalB})`;
        }
        
        case 'slope': {
            // Flat to steep (Solid steps: Green -> Yellow -> Orange -> Red)
            if (cell.slope < 15) return '#10b981';      // Stable green
            if (cell.slope < 28) return '#f59e0b';      // Moderate yellow
            if (cell.slope < 40) return '#f97316';      // Steep orange
            return '#ef4444';                           // Extreme slope red
        }
        
        case 'ndvi': {
            // Vegetation density (Solid Green scale)
            if (cell.ndvi < 0.2) return '#d6d3d1';      // Barren rock (gray)
            if (cell.ndvi < 0.4) return '#a7f3d0';      // Low grass
            if (cell.ndvi < 0.6) return '#34d399';      // Mixed shrubs
            return '#047857';                           // Dense forest (dark green)
        }
        
        case 'sar': {
            // Coherence Loss (Solid Purple scale representing ground deformation/landslides)
            if (cell.coherenceLoss < 0.15) return '#1e293b'; // Normal coherence
            if (cell.coherenceLoss < 0.4) return '#c084fc';  // Moderate decorrelation
            if (cell.coherenceLoss < 0.7) return '#a855f7';  // High change
            return '#7e22ce';                                // Landslide/extreme loss (deep purple)
        }
        
        case 'susceptibility': {
            // Landslide trigger probability (Solid hazard colors)
            if (cell.susceptibility < 0.20) return '#10b981'; // Very Low (Green)
            if (cell.susceptibility < 0.45) return '#fbbf24'; // Moderate (Yellow)
            if (cell.susceptibility < 0.70) return '#f97316'; // High (Orange)
            return '#ef4444';                                 // Very High (Red)
        }
    }
}

// --- Legend Rendering ---
function renderLegend() {
    const container = document.getElementById('legend-container');
    if (!container) return;
    
    let html = '';
    switch (state.activeLayer) {
        case 'terrain':
            html = `
                <div class="legend-item"><div class="legend-color-box" style="background-color: #64748b;"></div><span class="legend-desc">High Ridges (> 2000m)</span></div>
                <div class="legend-item"><div class="legend-color-box" style="background-color: #2d3748;"></div><span class="legend-desc">Mountain Foothills</span></div>
                <div class="legend-item"><div class="legend-color-box" style="background-color: #1a263a;"></div><span class="legend-desc">River Valleys (< 900m)</span></div>
            `;
            break;
        case 'slope':
            html = `
                <div class="legend-item"><div class="legend-color-box" style="background-color: #10b981;"></div><span class="legend-desc">Flat / Gentle (< 15°)</span></div>
                <div class="legend-item"><div class="legend-color-box" style="background-color: #f59e0b;"></div><span class="legend-desc">Moderate (15° - 28°)</span></div>
                <div class="legend-item"><div class="legend-color-box" style="background-color: #f97316;"></div><span class="legend-desc">Steep (28° - 40°)</span></div>
                <div class="legend-item"><div class="legend-color-box" style="background-color: #ef4444;"></div><span class="legend-desc">Cliffs / Esplanades (> 40°)</span></div>
            `;
            break;
        case 'ndvi':
            html = `
                <div class="legend-item"><div class="legend-color-box" style="background-color: #047857;"></div><span class="legend-desc">Dense Canopy / Forests (NDVI > 0.6)</span></div>
                <div class="legend-item"><div class="legend-color-box" style="background-color: #34d399;"></div><span class="legend-desc">Shrubs / Grass (NDVI 0.4 - 0.6)</span></div>
                <div class="legend-item"><div class="legend-color-box" style="background-color: #a7f3d0;"></div><span class="legend-desc">Sparse Vegetation (NDVI 0.2 - 0.4)</span></div>
                <div class="legend-item"><div class="legend-color-box" style="background-color: #d6d3d1;"></div><span class="legend-desc">Barren Rock / Water (NDVI < 0.2)</span></div>
            `;
            break;
        case 'sar':
            html = `
                <div class="legend-item"><div class="legend-color-box" style="background-color: #7e22ce;"></div><span class="legend-desc">Severe Decorrelation (Coherence Loss > 0.7)</span></div>
                <div class="legend-item"><div class="legend-color-box" style="background-color: #a855f7;"></div><span class="legend-desc">Moderate Decorrelation (0.4 - 0.7)</span></div>
                <div class="legend-item"><div class="legend-color-box" style="background-color: #c084fc;"></div><span class="legend-desc">Minor Decorrelation (0.15 - 0.4)</span></div>
                <div class="legend-item"><div class="legend-color-box" style="background-color: #1e293b;"></div><span class="legend-desc">Stable Surface (Coherence Loss < 0.15)</span></div>
            `;
            break;
        case 'susceptibility':
            html = `
                <div class="legend-item"><div class="legend-color-box" style="background-color: #ef4444;"></div><span class="legend-desc">Critical Landslide Risk (P > 70%)</span></div>
                <div class="legend-item"><div class="legend-color-box" style="background-color: #f97316;"></div><span class="legend-desc">High Risk (50% - 70%)</span></div>
                <div class="legend-item"><div class="legend-color-box" style="background-color: #fbbf24;"></div><span class="legend-desc">Moderate Risk (20% - 50%)</span></div>
                <div class="legend-item"><div class="legend-color-box" style="background-color: #10b981;"></div><span class="legend-desc">Stable / Low Risk (< 20%)</span></div>
            `;
            break;
    }
    container.innerHTML = html;
}

// --- Map Click/Hover Handlers ---
function getCellFromCoords(e) {
    const canvas = document.getElementById('map-canvas');
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    
    const size = state.cellSize;
    const offsetX = Math.floor((canvas.width - (state.gridCols * size)) / 2);
    const offsetY = Math.floor((canvas.height - (state.gridRows * size)) / 2);
    
    const col = Math.floor((x - offsetX) / size);
    const row = Math.floor((y - offsetY) / size);
    
    if (col >= 0 && col < state.gridCols && row >= 0 && row < state.gridRows) {
        return state.grid[row * state.gridCols + col];
    }
    return null;
}

function handleMapHover(e) {
    const cell = getCellFromCoords(e);
    if (cell !== state.hoveredCell) {
        state.hoveredCell = cell;
        drawMap();
        if (cell) {
            updateInspector(cell);
            if (window.traceNeuralNetwork) {
                window.traceNeuralNetwork(cell);
            }
        }
    }
}

function handleMapClick(e) {
    const cell = getCellFromCoords(e);
    if (cell) {
        // Set epicenter if simulation is not currently running
        if (!state.isSimulating) {
            state.epicenter = { col: cell.col, row: cell.row };
            document.getElementById('stat-epicenter').innerText = `Col ${cell.col}, Row ${cell.row}`;
            state.selectedCell = cell;
            drawMap();
            updateInspector(cell);
        } else {
            state.selectedCell = cell;
            drawMap();
            updateInspector(cell);
        }
    }
}

// --- Inspector Update ---
function updateInspector(cell) {
    document.getElementById('insp-coord').innerText = `Col ${cell.col}, Row ${cell.row}`;
    document.getElementById('insp-elev').innerText = `${cell.elevation} m`;
    document.getElementById('insp-slope').innerText = `${cell.slope}°`;
    
    // Aspect Cardinal mapping
    const cardinals = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'];
    const cardIndex = Math.round(cell.aspect / 45) % 8;
    document.getElementById('insp-aspect').innerText = `${cell.aspect}° (${cardinals[cardIndex]})`;
    
    document.getElementById('insp-ndvi').innerText = cell.ndvi.toFixed(2);
    document.getElementById('insp-sar').innerText = (1 - cell.coherenceLoss).toFixed(2);
    document.getElementById('insp-soil').innerText = `Class ${cell.soilClass} (Vs30 ${cell.vs30}m/s)`;
    
    document.getElementById('insp-pga').innerText = cell.pga.toFixed(3);
    document.getElementById('insp-arias').innerText = cell.arias.toFixed(2);
    
    const probPercent = (cell.susceptibility * 100).toFixed(1);
    const probLabel = document.getElementById('insp-probability');
    probLabel.innerText = `${probPercent}%`;
    
    // Color code the inspector probability label
    probLabel.parentElement.className = 'inspector-item danger';
    if (cell.susceptibility < 0.20) {
        probLabel.style.color = 'var(--color-success)';
    } else if (cell.susceptibility < 0.50) {
        probLabel.style.color = 'var(--color-warning)';
    } else {
        probLabel.style.color = 'var(--color-danger)';
    }
}

// --- Earthquake Wave Propagation Simulation ---
// Simulates wave spreading from epicenter, attenuating PGA/Arias and triggering landslides.
function triggerEarthquake() {
    if (!state.epicenter) {
        alert("Please set an epicenter first by clicking on the map or using 'Random Epicenter'.");
        return;
    }
    
    if (state.isSimulating) return;
    
    state.isSimulating = true;
    state.seismicWaveRadius = 0;
    
    // Reset ground motions
    state.grid.forEach(c => {
        c.pga = 0;
        c.arias = 0;
    });
    
    // Show wave trigger screen alert
    const alertOverlay = document.getElementById('eq-alert');
    alertOverlay.classList.add('show');
    
    setTimeout(() => {
        alertOverlay.classList.remove('show');
        runWaveStep();
    }, 1200);
}

function runWaveStep() {
    const size = state.gridCols;
    const maxRadius = Math.max(state.gridCols, state.gridRows) * 1.3;
    
    // Wave animation loops
    if (state.seismicWaveRadius < maxRadius) {
        state.seismicWaveRadius += 1.5;
        
        // Update cells within current wavefront
        updateWavefrontAttenuation(state.seismicWaveRadius);
        drawMap();
        updateDashboardMetrics();
        
        requestAnimationFrame(runWaveStep);
    } else {
        // Simulation finished
        state.isSimulating = false;
        state.seismicWaveRadius = 0;
        
        // Auto switch active layer to landslide susceptibility to show results
        state.activeLayer = 'susceptibility';
        
        // Activate buttons in layer selector
        const layerButtons = document.querySelectorAll('.layer-btn');
        layerButtons.forEach(btn => {
            if (btn.dataset.layer === 'susceptibility') {
                btn.classList.add('active');
            } else {
                btn.classList.remove('active');
            }
        });
        
        drawMap();
        renderLegend();
        updateDashboardMetrics();
        
        // Update inspector for selected cell
        if (state.hoveredCell) {
            updateInspector(state.hoveredCell);
        }
    }
}

// Estimates site-specific PGA and Arias Intensity based on the simulated wavefront
function updateWavefrontAttenuation(waveRadius) {
    const cols = state.gridCols;
    const rows = state.gridRows;
    const ep = state.epicenter;
    const mag = state.eqMagnitude;
    const depth = state.eqDepth;
    
    let currentMaxPga = 0.0;
    
    for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
            const idx = r * cols + c;
            const cell = state.grid[idx];
            
            // Grid-distance from epicenter
            const distCells = Math.sqrt((c - ep.col)**2 + (r - ep.row)**2);
            
            // Map cell distance to physical km (1 cell = 1.2km simulated scale)
            const distKm = distCells * 1.2;
            
            // Check if wave has reached this cell
            if (distCells <= waveRadius) {
                // Calculate PGA using Campbell/Joyner-Boore GMPE model
                // ln(pga) = -3.512 + 0.904*Mw - 1.328*ln(sqrt(Dist^2 + Depth^2)) - 0.0032*sqrt(Dist^2 + Depth^2)
                const r_eff = Math.sqrt(distKm**2 + depth**2);
                const ln_pga = -3.512 + 0.904 * mag - 1.328 * Math.log(r_eff) - 0.0032 * r_eff;
                let pga_rock = Math.exp(ln_pga);
                
                // Site Amplification
                let amp = 1.0;
                if (cell.soilClass === 'B') amp = 1.00;
                else if (cell.soilClass === 'C') amp = 1.25;
                else if (cell.soilClass === 'D') amp = 1.55;
                else if (cell.soilClass === 'E') amp = 1.85;
                
                cell.pga = Math.min(2.5, pga_rock * amp);
                if (cell.pga > currentMaxPga) currentMaxPga = cell.pga;
                
                // Arias Intensity Ia = pga^1.8 * site_amp * scaling
                cell.arias = Math.min(30.0, Math.pow(cell.pga, 1.8) * amp * 9.5);
                
                // --- FUSED DEEP LEARNING MODEL SIMULATION ---
                // Probability P of slope failure computed using inputs:
                // Slope, NDVI, Lithology, PGA, Arias
                
                const slopeTerm = Math.pow(cell.slope / 45.0, 2);   // steep slope is quadratic risk
                const vegTerm = 1.0 - cell.ndvi;                    // barren soils are unstable
                const seismicTerm = cell.pga * 2.3 + (cell.arias / 12.0); // shaking energy
                
                let lithologyMult = 1.0;
                if (cell.lithology === 0) lithologyMult = 0.5; // stable granite
                else if (cell.lithology === 2) lithologyMult = 1.5; // loose clay
                
                const score = (0.35 * slopeTerm + 0.1 * vegTerm + 0.55 * seismicTerm) * lithologyMult;
                
                // Sigmoid mapping
                cell.susceptibility = 1.0 / (1.0 + Math.exp(-3.0 * (score - 1.2)));
                
                // Trigger secondary physical effects: Coherence Loss on landslide cells
                if (cell.susceptibility > 0.70) {
                    // decorrelate SAR signal due to terrain displacement
                    cell.coherenceLoss = Math.min(0.95, cell.coherenceLoss + 0.4 * (cell.susceptibility - 0.6));
                    cell.coherencePost = cell.coherencePre - cell.coherenceLoss;
                    
                    // strip vegetation slightly
                    cell.ndvi = Math.max(0.02, cell.ndvi - 0.15 * cell.susceptibility);
                }
            }
        }
    }
    
    state.maxPGA = currentMaxPga;
    document.getElementById('stat-max-pga').innerText = `${state.maxPGA.toFixed(2)} g`;
}

// Reset the entire region to stable baseline state
function resetSimulation() {
    state.epicenter = null;
    state.seismicWaveRadius = 0;
    state.isSimulating = false;
    state.maxPGA = 0.0;
    
    document.getElementById('stat-epicenter').innerText = "Not Set";
    document.getElementById('stat-max-pga').innerText = "0.00 g";
    
    // Regenerate clean grid
    initTerrainGrid();
    drawMap();
    updateDashboardMetrics();
    
    if (state.hoveredCell) {
        updateInspector(state.hoveredCell);
    }
}

// --- Dashboards Statistics and Analytical Charts ---
function updateDashboardMetrics() {
    const cols = state.gridCols;
    const rows = state.gridRows;
    const totalCells = cols * rows;
    
    let criticalCount = 0;
    let counts = [0, 0, 0, 0]; // Low, Med, High, Crit
    
    state.grid.forEach(cell => {
        if (cell.susceptibility > 0.7) {
            criticalCount++;
            counts[3]++;
        } else if (cell.susceptibility > 0.5) {
            counts[2]++;
        } else if (cell.susceptibility > 0.2) {
            counts[1]++;
        } else {
            counts[0]++;
        }
    });
    
    // Critical area percentage
    const percentCrit = (criticalCount / totalCells) * 100;
    document.getElementById('metric-critical-area').innerText = `${percentCrit.toFixed(1)}%`;
    document.getElementById('metric-critical-fill').style.width = `${percentCrit}%`;
    
    // Update SVG/Canvas charts
    drawDistributionChart(counts);
    drawSHAPChart();
}

// Draw static solid-filled distribution chart for susceptibility classes
function drawDistributionChart(counts) {
    const canvas = document.getElementById('distribution-chart');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    const margin = 20;
    const chartHeight = canvas.height - margin * 1.5;
    const chartWidth = canvas.width - margin * 2;
    const barCount = 4;
    const spacing = 15;
    const barWidth = (chartWidth - (spacing * (barCount - 1))) / barCount;
    
    const maxVal = Math.max(...counts, 100);
    const labels = ["Low", "Moderate", "High", "Critical"];
    const colors = ["#10b981", "#fbbf24", "#f97316", "#ef4444"];
    
    ctx.font = '10px var(--font-sans)';
    ctx.textAlign = 'center';
    
    for (let i = 0; i < barCount; i++) {
        const val = counts[i];
        const pct = val / maxVal;
        const bHeight = chartHeight * pct;
        
        const x = margin + i * (barWidth + spacing);
        const y = canvas.height - margin - bHeight;
        
        // Draw bar (solid fills)
        ctx.fillStyle = colors[i];
        ctx.fillRect(x, y, barWidth, bHeight);
        
        // Draw value
        ctx.fillStyle = 'var(--text-primary)';
        ctx.fillText(val, x + barWidth / 2, y - 5);
        
        // Draw label
        ctx.fillStyle = 'var(--text-secondary)';
        ctx.fillText(labels[i], x + barWidth / 2, canvas.height - 5);
    }
}

// Draw SHAP-style horizontal feature importance chart based on selected cell
function drawSHAPChart() {
    const canvas = document.getElementById('shap-chart');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    const features = [
        { name: "PGA (Dynamic Shaking)", weight: 0.38, active: true },
        { name: "Slope Angle (Static)", weight: 0.28, active: true },
        { name: "Arias Intensity (Energy)", weight: 0.17, active: true },
        { name: "Lithology (Geology Class)", weight: 0.12, active: true },
        { name: "SAR Coherence Loss", weight: 0.08, active: true },
        { name: "NDVI (Vegetation Index)", weight: -0.05, active: true }
    ];
    
    // Sort features by absolute weight
    features.sort((a, b) => Math.abs(b.weight) - Math.abs(a.weight));
    
    const marginL = 135;
    const marginR = 25;
    const chartHeight = canvas.height - 10;
    const rowHeight = chartHeight / features.length;
    const scale = (canvas.width - marginL - marginR) / 0.5; // Max weight range is 0.5
    
    ctx.font = '10px var(--font-sans)';
    
    features.forEach((feat, idx) => {
        const y = idx * rowHeight + 10;
        
        // Draw feature label
        ctx.fillStyle = 'var(--text-secondary)';
        ctx.textAlign = 'left';
        ctx.fillText(feat.name, 5, y + 10);
        
        // Draw horizontal bar (solid color: blue for positive, red/orange for negative)
        const barLength = Math.abs(feat.weight) * scale;
        const startX = marginL;
        
        ctx.fillStyle = feat.weight > 0 ? '#3b82f6' : '#f97316';
        ctx.fillRect(startX, y + 2, barLength, rowHeight - 6);
        
        // Draw weight value text
        ctx.fillStyle = 'var(--text-primary)';
        ctx.textAlign = 'left';
        ctx.fillText((feat.weight > 0 ? "+" : "") + feat.weight.toFixed(2), startX + barLength + 5, y + 10);
    });
}
