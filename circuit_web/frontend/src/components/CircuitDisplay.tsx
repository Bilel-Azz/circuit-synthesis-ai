'use client';

import { Component } from '@/lib/api';
import { useState, useRef, useEffect } from 'react';

interface CircuitDisplayProps {
  components: Component[];
  title?: string;
}

// SVG component symbols
const SYMBOLS = {
  R: (
    <path
      d="M-25 0 L-18 0 L-14 -8 L-6 8 L2 -8 L10 8 L14 0 L25 0"
      fill="none"
      stroke="currentColor"
    />
  ),
  L: (
    <path
      d="M-25 0 L-18 0 Q-14 -10 -10 0 Q-6 -10 -2 0 Q2 -10 6 0 Q10 -10 14 0 L25 0"
      fill="none"
      stroke="currentColor"
    />
  ),
  C: (
    <g stroke="currentColor">
      <path d="M-25 0 L-4 0" />
      <path d="M25 0 L4 0" />
      <line x1="-4" y1="-12" x2="-4" y2="12" strokeWidth="3" />
      <line x1="4" y1="-12" x2="4" y2="12" strokeWidth="3" />
    </g>
  ),
  SOURCE: (
    <g>
      <circle cx="0" cy="0" r="18" fill="none" stroke="currentColor" strokeWidth="2" />
      <path d="M-8 0 Q-4 -6 0 0 T8 0" fill="none" stroke="currentColor" strokeWidth="2" />
    </g>
  )
};

export default function CircuitDisplay({ components, title }: CircuitDisplayProps) {
  const [transform, setTransform] = useState({ x: 0, y: 0, k: 1 });
  const [isDragging, setIsDragging] = useState(false);
  const [lastPos, setLastPos] = useState({ x: 0, y: 0 });
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    // Auto-fit on load if components change
    if (components.length > 0) {
      handleReset();
    }
  }, [components]);

  if (!components || components.length === 0) {
    return (
      <div className="p-12 bg-secondary/10 rounded-xl border border-dashed border-input flex flex-col items-center justify-center text-muted-foreground min-h-[300px]">
        <p className="font-bold text-lg">No circuit generated yet</p>
      </div>
    );
  }

  // 1. Analyze Topology
  const nodes = new Set<number>();
  components.forEach((c) => {
    nodes.add(c.node_a);
    nodes.add(c.node_b);
  });
  const uniqueNodes = Array.from(nodes).sort((a, b) => a - b);
  const signalNodes = uniqueNodes.filter(n => n !== 0);

  // Separate components by type
  const shuntComponents = components.filter(c => c.node_a === 0 || c.node_b === 0);
  const seriesComponents = components.filter(c => c.node_a !== 0 && c.node_b !== 0);

  // Find main path components (adjacent nodes) vs bypass (skip connections)
  const mainPathComps: Component[] = [];
  const bypassComps: Component[] = [];

  seriesComponents.forEach(c => {
    const idxA = signalNodes.indexOf(c.node_a);
    const idxB = signalNodes.indexOf(c.node_b);
    if (Math.abs(idxA - idxB) === 1) {
      mainPathComps.push(c);
    } else {
      bypassComps.push(c);
    }
  });

  // Layout Constants
  const GRID = 120;
  const PADDING = 100;
  const SCENE_WIDTH = Math.max(900, (signalNodes.length + 1) * GRID + PADDING * 2);
  const SCENE_HEIGHT = 500;
  const MAIN_Y = 200;  // Main horizontal line
  const BOT_Y = 380;   // Ground rail
  const SOURCE_X = PADDING;

  // Node positions
  const nodeX = new Map<number, number>();
  signalNodes.forEach((n, i) => {
    nodeX.set(n, PADDING + (i + 1) * GRID);
  });

  // Assign bypass components to different arc levels to avoid overlap
  const bypassLevels = new Map<Component, number>();
  const usedLevels: { minIdx: number; maxIdx: number; level: number }[] = [];

  bypassComps.forEach(comp => {
    const idxA = signalNodes.indexOf(comp.node_a);
    const idxB = signalNodes.indexOf(comp.node_b);
    const minIdx = Math.min(idxA, idxB);
    const maxIdx = Math.max(idxA, idxB);

    // Find a level that doesn't overlap
    let level = 1;
    while (usedLevels.some(u =>
      u.level === level && !(maxIdx < u.minIdx || minIdx > u.maxIdx)
    )) {
      level++;
    }

    bypassLevels.set(comp, level);
    usedLevels.push({ minIdx, maxIdx, level });
  });

  // --- Pan/Zoom Handlers ---
  const handleWheel = (e: React.WheelEvent) => {
    e.preventDefault();
    e.stopPropagation();

    const scaleFactor = 1.1;
    const delta = -e.deltaY;
    const newScale = delta > 0 ? transform.k * scaleFactor : transform.k / scaleFactor;

    // Limit zoom
    if (newScale < 0.2 || newScale > 5) return;

    // Zoom towards center (simplified) or mouse pointer
    // For simplicity, zoom center
    setTransform(prev => ({ ...prev, k: newScale }));
  };

  const handleMouseDown = (e: React.MouseEvent) => {
    setIsDragging(true);
    setLastPos({ x: e.clientX, y: e.clientY });
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (!isDragging) return;
    const dx = e.clientX - lastPos.x;
    const dy = e.clientY - lastPos.y;
    setTransform(prev => ({ ...prev, x: prev.x + dx, y: prev.y + dy }));
    setLastPos({ x: e.clientX, y: e.clientY });
  };

  const handleMouseUp = () => setIsDragging(false);

  const handleReset = () => {
    // Fit to screen logic
    if (!containerRef.current) return;
    const { width, height } = containerRef.current.getBoundingClientRect();
    const scaleX = width / SCENE_WIDTH;
    const scaleY = height / SCENE_HEIGHT;
    const minScale = Math.min(scaleX, scaleY) * 0.9; // 90% fit

    const startX = (width - SCENE_WIDTH * minScale) / 2;
    const startY = (height - SCENE_HEIGHT * minScale) / 2;

    setTransform({ x: startX, y: startY, k: minScale });
  };

  return (
    <div className="space-y-6">
      {title && (
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="w-1.5 h-6 bg-primary rounded-full shadow-[0_0_10px_rgba(59,130,246,0.5)]"></div>
            <h3 className="text-xl font-black tracking-tight text-foreground">{title}</h3>
          </div>

          <div className="flex gap-2">
            <button
              onClick={() => {
                // Generate SPICE netlist
                const formatValue = (val: number, type: string) => {
                  if (type === 'R') {
                    if (val >= 1e6) return `${(val/1e6).toFixed(3)}Meg`;
                    if (val >= 1e3) return `${(val/1e3).toFixed(3)}k`;
                    return `${val.toFixed(3)}`;
                  } else if (type === 'L') {
                    if (val >= 1) return `${val.toFixed(6)}`;
                    if (val >= 1e-3) return `${(val*1e3).toFixed(3)}m`;
                    if (val >= 1e-6) return `${(val*1e6).toFixed(3)}u`;
                    return `${(val*1e9).toFixed(3)}n`;
                  } else if (type === 'C') {
                    if (val >= 1e-6) return `${(val*1e6).toFixed(3)}u`;
                    if (val >= 1e-9) return `${(val*1e9).toFixed(3)}n`;
                    return `${(val*1e12).toFixed(3)}p`;
                  }
                  return val.toString();
                };

                let spice = '* Circuit generated by Circuit Synthesis AI\n';
                spice += '* https://github.com/circuit-synthesis\n\n';
                spice += '* AC source at node 1\n';
                spice += 'VAC 1 0 AC 1\n\n';
                spice += '* Components\n';

                components.forEach((c, i) => {
                  const name = `${c.type}${i + 1}`;
                  const val = formatValue(c.value, c.type);
                  spice += `${name} ${c.node_a} ${c.node_b} ${val}\n`;
                });

                spice += '\n* AC Analysis (10Hz to 10MHz)\n';
                spice += '.AC DEC 100 10 10Meg\n';
                spice += '.END\n';

                const blob = new Blob([spice], { type: 'text/plain' });
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url; a.download = 'circuit.cir';
                document.body.appendChild(a); a.click(); document.body.removeChild(a); URL.revokeObjectURL(url);
              }}
              className="px-3 py-1.5 text-xs font-bold rounded-lg border border-primary/50 bg-primary/10 text-primary hover:bg-primary/20 transition-colors"
            >
              Export SPICE
            </button>
            <button
              onClick={() => {
                navigator.clipboard.writeText(JSON.stringify(components, null, 2));
                alert('Circuit JSON copied to clipboard!');
              }}
              className="px-3 py-1.5 text-xs font-bold rounded-lg border border-border bg-background hover:bg-secondary transition-colors"
            >
              Copy JSON
            </button>
            <button
              onClick={() => {
                const blob = new Blob([JSON.stringify(components, null, 2)], { type: 'application/json' });
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url; a.download = 'circuit.json';
                document.body.appendChild(a); a.click(); document.body.removeChild(a); URL.revokeObjectURL(url);
              }}
              className="px-3 py-1.5 text-xs font-bold rounded-lg border border-border bg-background hover:bg-secondary transition-colors"
            >
              Download
            </button>
          </div>
        </div>
      )}

      {/* Schematic Container - Force light mode for visibility */}
      <div
        ref={containerRef}
        className="bg-white rounded-2xl border border-gray-200 shadow-2xl relative overflow-hidden group h-[500px] cursor-grab active:cursor-grabbing"
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
        onWheel={handleWheel}
      >
        <div className="absolute inset-0 bg-[linear-gradient(to_right,#80808015_1px,transparent_1px),linear-gradient(to_bottom,#80808015_1px,transparent_1px)] bg-[size:20px_20px] pointer-events-none" />

        {/* Controls Overlay */}
        <div className="absolute bottom-4 right-4 flex flex-col gap-2 bg-background/80 backdrop-blur border border-border rounded-lg p-1.5 shadow-lg z-10">
          <button onClick={() => setTransform(t => ({ ...t, k: t.k * 1.2 }))} className="p-2 hover:bg-secondary rounded-md" title="Zoom In">+</button>
          <button onClick={() => setTransform(t => ({ ...t, k: t.k / 1.2 }))} className="p-2 hover:bg-secondary rounded-md" title="Zoom Out">-</button>
          <button onClick={handleReset} className="p-2 hover:bg-secondary rounded-md" title="Reset View">⟲</button>
        </div>

        <svg
          width="100%"
          height="100%"
          className="select-none"
        >
          <defs>
            <filter id="glow" x="-20%" y="-20%" width="140%" height="140%">
              <feGaussianBlur stdDeviation="2" result="blur" />
              <feComposite in="SourceGraphic" in2="blur" operator="over" />
            </filter>
          </defs>

          <g transform={`translate(${transform.x}, ${transform.y}) scale(${transform.k})`}>
            {/* --- 1. Draw Main Rails --- */}

            {/* Source Symbol */}
            <g transform={`translate(${SOURCE_X}, ${MAIN_Y})`} className="text-primary">
              {SYMBOLS.SOURCE}
              <text x="-35" y="5" textAnchor="end" className="text-xs font-bold font-mono fill-muted-foreground">VAC</text>
            </g>

            {/* Wire: Source to First Node */}
            <line
              x1={SOURCE_X + 18} y1={MAIN_Y}
              x2={nodeX.get(signalNodes[0])!} y2={MAIN_Y}
              stroke="currentColor" strokeWidth="2.5" className="text-foreground/70"
            />

            {/* Main horizontal line connecting nodes */}
            <line
              x1={nodeX.get(signalNodes[0])!} y1={MAIN_Y}
              x2={nodeX.get(signalNodes[signalNodes.length - 1])!} y2={MAIN_Y}
              stroke="currentColor" strokeWidth="2.5" className="text-foreground/70"
            />

            {/* Ground rail */}
            <line
              x1={SOURCE_X} y1={BOT_Y}
              x2={nodeX.get(signalNodes[signalNodes.length - 1])! + 40} y2={BOT_Y}
              stroke="currentColor" strokeWidth="2.5" className="text-foreground/70"
            />

            {/* Source to ground connection */}
            <line x1={SOURCE_X} y1={MAIN_Y + 18} x2={SOURCE_X} y2={BOT_Y}
              stroke="currentColor" strokeWidth="2.5" className="text-foreground/70" />

            {/* --- 2. Draw Shunt Components (to ground) --- */}
            {shuntComponents.map((comp, idx) => {
              const signalNode = comp.node_a === 0 ? comp.node_b : comp.node_a;
              const x = nodeX.get(signalNode)!;
              const cy = (MAIN_Y + BOT_Y) / 2;

              return (
                <g key={`shunt-${idx}`}>
                  {/* Vertical wires */}
                  <line x1={x} y1={MAIN_Y} x2={x} y2={cy - 25} stroke="currentColor" strokeWidth="2.5" className="text-foreground/70" />
                  <line x1={x} y1={cy + 25} x2={x} y2={BOT_Y} stroke="currentColor" strokeWidth="2.5" className="text-foreground/70" />
                  {/* Component */}
                  <g transform={`translate(${x}, ${cy}) rotate(90)`}>
                    <g className="text-blue-500 text-blue-500 stroke-[2.5px]">
                      {SYMBOLS[comp.type as 'R' | 'L' | 'C']}
                    </g>
                  </g>
                  {/* Label */}
                  <text x={x + 30} y={cy + 4} className="text-[11px] font-bold font-mono fill-blue-600 fill-blue-600">{comp.formatted_value}</text>
                </g>
              );
            })}

            {/* --- 3. Draw Main Path Components (between adjacent nodes) --- */}
            {mainPathComps.map((comp, idx) => {
              const xA = nodeX.get(comp.node_a)!;
              const xB = nodeX.get(comp.node_b)!;
              const cx = (xA + xB) / 2;

              return (
                <g key={`main-${idx}`}>
                  <g transform={`translate(${cx}, ${MAIN_Y})`}>
                    <g className="text-orange-500 text-orange-500 stroke-[2.5px]">
                      {SYMBOLS[comp.type as 'R' | 'L' | 'C']}
                    </g>
                  </g>
                  {/* Label above */}
                  <text x={cx} y={MAIN_Y - 20} textAnchor="middle" className="text-[11px] font-bold font-mono fill-orange-600 fill-orange-600">{comp.formatted_value}</text>
                </g>
              );
            })}

            {/* --- 4. Draw Bypass Components (skip connections) --- */}
            {bypassComps.map((comp, idx) => {
              const xA = nodeX.get(comp.node_a)!;
              const xB = nodeX.get(comp.node_b)!;
              const level = bypassLevels.get(comp) || 1;
              const arcY = MAIN_Y - 50 - (level - 1) * 45;  // Stack arcs at different heights
              const cx = (xA + xB) / 2;

              return (
                <g key={`bypass-${idx}`}>
                  {/* Arc wires using bezier curves */}
                  <path
                    d={`M ${xA} ${MAIN_Y}
                        C ${xA} ${arcY}, ${xA + 30} ${arcY}, ${cx - 25} ${arcY}`}
                    fill="none" stroke="currentColor" strokeWidth="2" className="text-foreground/50"
                  />
                  <path
                    d={`M ${cx + 25} ${arcY}
                        C ${xB - 30} ${arcY}, ${xB} ${arcY}, ${xB} ${MAIN_Y}`}
                    fill="none" stroke="currentColor" strokeWidth="2" className="text-foreground/50"
                  />
                  {/* Component */}
                  <g transform={`translate(${cx}, ${arcY})`}>
                    <g className="text-green-600 text-green-600 stroke-[2.5px]">
                      {SYMBOLS[comp.type as 'R' | 'L' | 'C']}
                    </g>
                  </g>
                  {/* Label */}
                  <text x={cx} y={arcY - 15} textAnchor="middle" className="text-[11px] font-bold font-mono fill-green-600 fill-green-600">{comp.formatted_value}</text>
                </g>
              );
            })}

            {/* --- 5. Draw Nodes --- */}
            {signalNodes.map((n) => (
              <g key={n} transform={`translate(${nodeX.get(n)}, ${MAIN_Y})`}>
                <circle r="5" className="fill-background stroke-[2.5px] stroke-foreground" />
                <text y={25} textAnchor="middle" className="text-[10px] font-bold fill-muted-foreground">N{n}</text>
              </g>
            ))}

            {/* Ground Symbol */}
            <g transform={`translate(${nodeX.get(signalNodes[signalNodes.length - 1])! + 40}, ${BOT_Y})`}>
              <line x1="0" y1="0" x2="0" y2="8" stroke="currentColor" strokeWidth="2.5" className="text-foreground/70" />
              <line x1="-12" y1="8" x2="12" y2="8" stroke="currentColor" strokeWidth="2.5" className="text-foreground/70" />
              <line x1="-7" y1="13" x2="7" y2="13" stroke="currentColor" strokeWidth="2" className="text-foreground/70" />
              <line x1="-3" y1="18" x2="3" y2="18" stroke="currentColor" strokeWidth="1.5" className="text-foreground/70" />
            </g>
          </g>
        </svg>
      </div>

      {/* Net list */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
        {components.map((comp, idx) => (
          <div key={idx} className="flex items-center gap-3 p-4 rounded-xl border border-border bg-card/40 shadow-sm">
            <div className="font-black text-lg text-primary">{comp.type}</div>
            <div className="text-sm font-mono">{comp.formatted_value}</div>
          </div>
        ))}
      </div>
    </div>
  );
}
