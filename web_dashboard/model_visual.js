/**
 * SILSM - Deep Learning Architecture Visualizer
 * Draws a clean representation of the CNN + MLP Fusion Network
 * and animates information tracing based on selected grid cell parameters.
 */

let visualizerState = {
    canvas: null,
    ctx: null,
    activeCell: null,
    nodes: [],
    connections: [],
    animationFrameId: null,
    signalOffset: 0
};

// Initializes the visualizer canvas coordinates and layers
window.initModelVisualizer = function() {
    const canvas = document.getElementById('model-canvas');
    if (!canvas) return;
    
    // Set width from container client size
    const container = canvas.parentElement;
    canvas.width = container.clientWidth;
    canvas.height = container.clientHeight;
    
    visualizerState.canvas = canvas;
    visualizerState.ctx = canvas.getContext('2d');
    
    buildNeuralLayout();
    startAnimationLoop();
    
    // Listen for resize
    window.addEventListener('resize', () => {
        canvas.width = container.clientWidth;
        canvas.height = container.clientHeight;
        buildNeuralLayout();
    });
};

// Rebuilds the layer positions based on canvas width
function buildNeuralLayout() {
    const canvas = visualizerState.canvas;
    if (!canvas) return;
    
    const w = canvas.width;
    const h = canvas.height;
    const ctx = visualizerState.ctx;
    
    visualizerState.nodes = [];
    visualizerState.connections = [];
    
    // Define vertical layers
    const xInputSpatial = w * 0.12;
    const xInputSeismic = w * 0.12;
    
    const xCnnBlock = w * 0.32;
    const xMlpBlock = w * 0.32;
    
    const xFusionNode = w * 0.60;
    const xClassifier = w * 0.80;
    const xOutput = w * 0.93;
    
    // 1. Spatial Inputs Nodes (Static + Satellite)
    const spatialLabels = ["Slope", "DEM", "NDVI", "SAR", "Lithology"];
    const sNodeCount = spatialLabels.length;
    const sSpacing = h * 0.45 / (sNodeCount + 1);
    const sStartY = 10;
    
    const spatialNodes = [];
    for (let i = 0; i < sNodeCount; i++) {
        const node = {
            id: `in_sp_${i}`,
            x: xInputSpatial,
            y: sStartY + (i + 1) * sSpacing,
            label: spatialLabels[i],
            type: 'input',
            activity: 0.1
        };
        visualizerState.nodes.push(node);
        spatialNodes.push(node);
    }
    
    // 2. Seismic Inputs Nodes (Dynamic)
    const seismicLabels = ["PGA", "Arias (Ia)", "Vs30"];
    const seisNodeCount = seismicLabels.length;
    const seisSpacing = h * 0.40 / (seisNodeCount + 1);
    const seisStartY = h * 0.50;
    
    const seismicNodes = [];
    for (let i = 0; i < seisNodeCount; i++) {
        const node = {
            id: `in_se_${i}`,
            x: xInputSeismic,
            y: seisStartY + (i + 1) * seisSpacing,
            label: seismicLabels[i],
            type: 'input',
            activity: 0.1
        };
        visualizerState.nodes.push(node);
        seismicNodes.push(node);
    }
    
    // 3. Encoder Blocks (Represented as visual boxes/nodes)
    const cnnNode = {
        id: 'enc_cnn',
        x: xCnnBlock,
        y: h * 0.28,
        label: 'Spatial 2D CNN',
        type: 'encoder',
        activity: 0.2
    };
    const mlpNode = {
        id: 'enc_mlp',
        x: xMlpBlock,
        y: h * 0.70,
        label: 'Seismic MLP',
        type: 'encoder',
        activity: 0.2
    };
    
    visualizerState.nodes.push(cnnNode);
    visualizerState.nodes.push(mlpNode);
    
    // 4. Fusion Node (Concatenation layer)
    const fusionNode = {
        id: 'fusion',
        x: xFusionNode,
        y: h * 0.5,
        label: 'Multi-Modal Fusion',
        type: 'fusion',
        activity: 0.15
    };
    visualizerState.nodes.push(fusionNode);
    
    // 5. Classifier MLP Layers
    const classifierNode = {
        id: 'classifier',
        x: xClassifier,
        y: h * 0.5,
        label: 'Dense Head',
        type: 'classifier',
        activity: 0.1
    };
    visualizerState.nodes.push(classifierNode);
    
    // 6. Final Sigmoid output node
    const outputNode = {
        id: 'output',
        x: xOutput,
        y: h * 0.5,
        label: 'Trigger Prob (P)',
        type: 'output',
        activity: 0.05
    };
    visualizerState.nodes.push(outputNode);
    
    // --- Connect Layers ---
    // Connect Spatial inputs -> CNN Block
    spatialNodes.forEach(sn => {
        visualizerState.connections.push({ from: sn, to: cnnNode, weight: 0.5 });
    });
    
    // Connect Seismic inputs -> MLP Block
    seismicNodes.forEach(sen => {
        visualizerState.connections.push({ from: sen, to: mlpNode, weight: 0.5 });
    });
    
    // Connect CNN and MLP encoders -> Fusion Node
    visualizerState.connections.push({ from: cnnNode, to: fusionNode, weight: 0.8 });
    visualizerState.connections.push({ from: mlpNode, to: fusionNode, weight: 0.8 });
    
    // Connect Fusion -> Classifier Head -> Output
    visualizerState.connections.push({ from: fusionNode, to: classifierNode, weight: 0.9 });
    visualizerState.connections.push({ from: classifierNode, to: outputNode, weight: 0.9 });
}

// Traces neural weights and activates pathways according to hovered cell features
window.traceNeuralNetwork = function(cell) {
    if (!cell) return;
    visualizerState.activeCell = cell;
    
    // Update individual input nodes activations based on physical values
    visualizerState.nodes.forEach(node => {
        switch (node.id) {
            case 'in_sp_0': // Slope (0 - 50 deg normalized)
                node.activity = Math.min(1.0, cell.slope / 45.0);
                break;
            case 'in_sp_1': // DEM Elevation (500 - 2500 normalized)
                node.activity = (cell.elevation - 500) / 2000.0;
                break;
            case 'in_sp_2': // NDVI (opposite: low vegetation = higher activity of risk channel)
                node.activity = 1.0 - cell.ndvi;
                break;
            case 'in_sp_3': // SAR coherence loss (0.05 to 0.9 normalized)
                node.activity = cell.coherenceLoss;
                break;
            case 'in_sp_4': // Lithology (index 2 clay = high activity risk)
                node.activity = cell.lithology === 2 ? 0.9 : (cell.lithology === 1 ? 0.5 : 0.2);
                break;
                
            case 'in_se_0': // PGA (0 to 1.5g normalized)
                node.activity = Math.min(1.0, cell.pga / 1.2);
                break;
            case 'in_se_1': // Arias (0 to 30 normalized)
                node.activity = Math.min(1.0, cell.arias / 20.0);
                break;
            case 'in_se_2': // Vs30 (softer soil = higher vulnerability channel)
                node.activity = cell.soilClass === 'E' ? 0.9 : (cell.soilClass === 'D' ? 0.7 : 0.3);
                break;
                
            case 'enc_cnn':
                // average of spatial inputs
                const spNodeIds = ['in_sp_0', 'in_sp_1', 'in_sp_2', 'in_sp_3', 'in_sp_4'];
                const avgSp = visualizerState.nodes
                    .filter(n => spNodeIds.includes(n.id))
                    .reduce((sum, n) => sum + n.activity, 0) / 5;
                node.activity = avgSp;
                break;
                
            case 'enc_mlp':
                // average of seismic inputs
                const seNodeIds = ['in_se_0', 'in_se_1', 'in_se_2'];
                const avgSe = visualizerState.nodes
                    .filter(n => seNodeIds.includes(n.id))
                    .reduce((sum, n) => sum + n.activity, 0) / 3;
                node.activity = avgSe;
                break;
                
            case 'fusion':
                // combined spatial-seismic
                const cnnAct = visualizerState.nodes.find(n => n.id === 'enc_cnn').activity;
                const mlpAct = visualizerState.nodes.find(n => n.id === 'enc_mlp').activity;
                node.activity = (cnnAct * 0.4 + mlpAct * 0.6);
                break;
                
            case 'classifier':
                node.activity = visualizerState.nodes.find(n => n.id === 'fusion').activity * 0.9;
                break;
                
            case 'output':
                node.activity = cell.susceptibility;
                break;
        }
    });
};

// Starts visualizer rendering loop
function startAnimationLoop() {
    function animate() {
        drawVisualizer();
        visualizerState.signalOffset = (visualizerState.signalOffset + 0.015) % 1.0;
        visualizerState.animationFrameId = requestAnimationFrame(animate);
    }
    animate();
}

// Renders the visualizer block elements onto the bottom canvas
function drawVisualizer() {
    const canvas = visualizerState.canvas;
    const ctx = visualizerState.ctx;
    if (!canvas || !ctx) return;
    
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // 1. Draw connections
    visualizerState.connections.forEach(conn => {
        const fromAct = conn.from.activity || 0.1;
        const toAct = conn.to.activity || 0.1;
        const activePct = (fromAct + toAct) / 2.0;
        
        // Solid line color based on activation intensity
        ctx.strokeStyle = activePct > 0.6 
            ? 'rgba(239, 68, 68, 0.4)' // Red for high danger propagation
            : (activePct > 0.3 
                ? 'rgba(249, 115, 22, 0.3)' // Orange for moderate
                : 'rgba(75, 85, 99, 0.25)'); // Dark gray for low activity
        
        ctx.lineWidth = 1 + activePct * 2;
        ctx.beginPath();
        ctx.moveTo(conn.from.x, conn.from.y);
        ctx.lineTo(conn.to.x, conn.to.y);
        ctx.stroke();
        
        // Draw animating data transmission dots along the connections
        const xDiff = conn.to.x - conn.from.x;
        const yDiff = conn.to.y - conn.from.y;
        
        const dotX = conn.from.x + xDiff * visualizerState.signalOffset;
        const dotY = conn.from.y + yDiff * visualizerState.signalOffset;
        
        ctx.fillStyle = activePct > 0.6 ? '#ef4444' : (activePct > 0.3 ? '#f97316' : '#94a3b8');
        ctx.beginPath();
        ctx.arc(dotX, dotY, 2.5, 0, 2 * Math.PI);
        ctx.fill();
    });
    
    // 2. Draw nodes
    visualizerState.nodes.forEach(node => {
        const act = node.activity || 0.1;
        
        // Node circle color matching activation
        let fillStyle = '#1e293b'; // Normal dark gray
        let borderStyle = '#4b5563';
        
        if (node.type === 'output') {
            // Output changes green/red
            fillStyle = act > 0.70 ? '#ef4444' : (act > 0.30 ? '#f97316' : '#10b981');
            borderStyle = '#ffffff';
        } else {
            // Encoder / hidden nodes
            if (act > 0.6) {
                fillStyle = 'rgba(239, 68, 68, 0.9)'; // Red highlight
                borderStyle = '#ef4444';
            } else if (act > 0.3) {
                fillStyle = 'rgba(249, 115, 22, 0.8)'; // Orange highlight
                borderStyle = '#f97316';
            } else {
                fillStyle = '#1e2530'; // Normal baseline
                borderStyle = '#2b3548';
            }
        }
        
        // Draw node body
        ctx.fillStyle = fillStyle;
        ctx.strokeStyle = borderStyle;
        ctx.lineWidth = node.type === 'output' ? 2 : 1.5;
        
        ctx.beginPath();
        if (node.type === 'encoder') {
            // Draw rectangle for major processing blocks (CNN / MLP encoders)
            const boxW = 100;
            const boxH = 26;
            ctx.roundRect(node.x - boxW/2, node.y - boxH/2, boxW, boxH, 4);
            ctx.fill();
            ctx.stroke();
            
            // Text label
            ctx.fillStyle = '#ffffff';
            ctx.font = '9px var(--font-sans)';
            ctx.textAlign = 'center';
            ctx.fillText(node.label, node.x, node.y + 3);
        } else {
            // Circular nodes
            const r = node.type === 'output' ? 12 : (node.type === 'fusion' ? 9 : 6);
            ctx.arc(node.x, node.y, r, 0, 2 * Math.PI);
            ctx.fill();
            ctx.stroke();
            
            // Text label for inputs or outputs
            if (node.type === 'input') {
                ctx.fillStyle = 'var(--text-secondary)';
                ctx.font = '9px var(--font-sans)';
                ctx.textAlign = 'right';
                ctx.fillText(node.label, node.x - 10, node.y + 3);
            } else if (node.type === 'output') {
                ctx.fillStyle = '#ffffff';
                ctx.font = '10px var(--font-sans)';
                ctx.textAlign = 'left';
                ctx.fillText(`${node.label}: ${(act*100).toFixed(0)}%`, node.x + 16, node.y + 3);
            } else if (node.type === 'fusion') {
                ctx.fillStyle = 'var(--text-secondary)';
                ctx.font = '9px var(--font-sans)';
                ctx.textAlign = 'center';
                ctx.fillText('Fusion', node.x, node.y - 12);
            } else if (node.type === 'classifier') {
                ctx.fillStyle = 'var(--text-secondary)';
                ctx.font = '9px var(--font-sans)';
                ctx.textAlign = 'center';
                ctx.fillText('FC Layer', node.x, node.y - 12);
            }
        }
    });
}
