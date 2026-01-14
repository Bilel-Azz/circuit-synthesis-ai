'use client';

import { useState, useEffect } from 'react';
import { Config, getConfig } from '@/lib/api';

interface ImpedanceInputProps {
  onSubmit: (magnitude: number[], phase: number[]) => void;
  loading?: boolean;
}

// Sample impedance curves - Complex RLC circuits that showcase model capabilities
const SAMPLE_CURVES = [
  {
    name: 'RLC Serie Resonant',
    desc: 'R=100Ω + L=1mH + C=100nF (f0 ≈ 16kHz)',
    icon: '📊',
    generate: (freqs: number[]) => {
      const R = 100;
      const L = 1e-3;
      const C = 100e-9;
      return freqs.map((f) => {
        const omega = 2 * Math.PI * f;
        const Zl = omega * L;
        const Zc = 1 / (omega * C);
        const Zreal = R;
        const Zimag = Zl - Zc;
        const Zmag = Math.sqrt(Zreal * Zreal + Zimag * Zimag);
        const Zphase = Math.atan2(Zimag, Zreal);
        return { magnitude: Math.log10(Zmag), phase: Zphase };
      });
    },
  },
  {
    name: 'Tank LC Anti-Resonant',
    desc: 'R=50Ω + (L=10mH || C=1µF)',
    icon: '📈',
    generate: (freqs: number[]) => {
      const Rs = 50;
      const Rp = 10;
      const L = 10e-3;
      const C = 1e-6;
      return freqs.map((f) => {
        const omega = 2 * Math.PI * f;
        const Zl = omega * L;
        const Zl_mag_sq = Rp * Rp + Zl * Zl;
        const Yl_real = Rp / Zl_mag_sq;
        const Yl_imag = -Zl / Zl_mag_sq;
        const Y_real = Yl_real;
        const Y_imag = Yl_imag + omega * C;
        const Y_mag = Math.sqrt(Y_real * Y_real + Y_imag * Y_imag);
        const Zp_real = Y_real / (Y_mag * Y_mag);
        const Zp_imag = -Y_imag / (Y_mag * Y_mag);
        const Zt_real = Rs + Zp_real;
        const Zt_imag = Zp_imag;
        const Zmag = Math.sqrt(Zt_real * Zt_real + Zt_imag * Zt_imag);
        const Zphase = Math.atan2(Zt_imag, Zt_real);
        return { magnitude: Math.log10(Zmag), phase: Zphase };
      });
    },
  },
  {
    name: 'Passe-Bande RLC',
    desc: 'R=200Ω + L=5mH + C=500nF',
    icon: '🎚️',
    generate: (freqs: number[]) => {
      const R = 200;
      const L = 5e-3;
      const C = 500e-9;
      return freqs.map((f) => {
        const omega = 2 * Math.PI * f;
        const Zreal = R;
        const Zimag = omega * L - 1 / (omega * C);
        const Zmag = Math.sqrt(Zreal * Zreal + Zimag * Zimag);
        const Zphase = Math.atan2(Zimag, Zreal);
        return { magnitude: Math.log10(Zmag), phase: Zphase };
      });
    },
  },
  {
    name: 'Double Resonance',
    desc: 'RLC serie + RL parallele (2 pics)',
    icon: '〰️',
    generate: (freqs: number[]) => {
      const R1 = 50, R2 = 500;
      const L1 = 2e-3, L2 = 10e-3;
      const C1 = 100e-9;
      return freqs.map((f) => {
        const omega = 2 * Math.PI * f;
        const Z1_real = R1;
        const Z1_imag = omega * L1 - 1 / (omega * C1);
        const Zl2 = omega * L2;
        const Y2_real = 1 / R2;
        const Y2_imag = -1 / Zl2;
        const Y2_mag = Math.sqrt(Y2_real * Y2_real + Y2_imag * Y2_imag);
        const Z2_real = Y2_real / (Y2_mag * Y2_mag);
        const Z2_imag = -Y2_imag / (Y2_mag * Y2_mag);
        const Zt_real = Z1_real + Z2_real;
        const Zt_imag = Z1_imag + Z2_imag;
        const Zmag = Math.sqrt(Zt_real * Zt_real + Zt_imag * Zt_imag);
        const Zphase = Math.atan2(Zt_imag, Zt_real);
        return { magnitude: Math.log10(Zmag), phase: Zphase };
      });
    },
  },
  {
    name: 'Ladder RLC 3 etages',
    desc: 'R-L-C-R-L-C (filtre complexe)',
    icon: '🪜',
    generate: (freqs: number[]) => {
      const R1 = 100, R2 = 200;
      const L1 = 1e-3, L2 = 2e-3;
      const C1 = 100e-9, C2 = 220e-9;
      return freqs.map((f) => {
        const omega = 2 * Math.PI * f;
        const Z2_real = R2;
        const Z2_imag = omega * L2 - 1 / (omega * C2);
        const Yc1 = omega * C1;
        const Y2_real = Z2_real / (Z2_real * Z2_real + Z2_imag * Z2_imag);
        const Y2_imag = -Z2_imag / (Z2_real * Z2_real + Z2_imag * Z2_imag);
        const Yp_real = Y2_real;
        const Yp_imag = Y2_imag + Yc1;
        const Yp_mag = Math.sqrt(Yp_real * Yp_real + Yp_imag * Yp_imag);
        const Zp_real = Yp_real / (Yp_mag * Yp_mag);
        const Zp_imag = -Yp_imag / (Yp_mag * Yp_mag);
        const Zt_real = R1 + Zp_real;
        const Zt_imag = omega * L1 + Zp_imag;
        const Zmag = Math.sqrt(Zt_real * Zt_real + Zt_imag * Zt_imag);
        const Zphase = Math.atan2(Zt_imag, Zt_real);
        return { magnitude: Math.log10(Zmag), phase: Zphase };
      });
    },
  },
  {
    name: 'Notch Filter',
    desc: 'R + LC parallele (encoche a 5kHz)',
    icon: '🚫',
    generate: (freqs: number[]) => {
      const Rs = 100;
      const L = 10e-3;
      const C = 100e-9;
      return freqs.map((f) => {
        const omega = 2 * Math.PI * f;
        const Yl_imag = -1 / (omega * L);
        const Yc_imag = omega * C;
        const Y_imag = Yl_imag + Yc_imag;
        const Y_mag = Math.abs(Y_imag);
        const Zp = Y_mag > 1e-10 ? 1 / Y_mag : 1e10;
        const Zp_phase = Y_imag > 0 ? -Math.PI / 2 : Math.PI / 2;
        const Zp_real = Zp * Math.cos(Zp_phase);
        const Zp_imag = Zp * Math.sin(Zp_phase);
        const Zt_real = Rs + Zp_real;
        const Zt_imag = Zp_imag;
        const Zmag = Math.sqrt(Zt_real * Zt_real + Zt_imag * Zt_imag);
        const Zphase = Math.atan2(Zt_imag, Zt_real);
        return { magnitude: Math.log10(Zmag), phase: Zphase };
      });
    },
  },
  {
    name: 'Circuit 5 Composants',
    desc: 'R1-L1-C1 serie + R2||C2 shunt',
    icon: '🔀',
    generate: (freqs: number[]) => {
      const R1 = 150, R2 = 1000;
      const L1 = 3e-3;
      const C1 = 150e-9, C2 = 470e-9;
      return freqs.map((f) => {
        const omega = 2 * Math.PI * f;
        const Yc2 = omega * C2;
        const Yr2 = 1 / R2;
        const Ysh_mag = Math.sqrt(Yr2 * Yr2 + Yc2 * Yc2);
        const Zsh_real = Yr2 / (Ysh_mag * Ysh_mag);
        const Zsh_imag = -Yc2 / (Ysh_mag * Ysh_mag);
        const Zt_real = R1 + Zsh_real;
        const Zt_imag = omega * L1 - 1 / (omega * C1) + Zsh_imag;
        const Zmag = Math.sqrt(Zt_real * Zt_real + Zt_imag * Zt_imag);
        const Zphase = Math.atan2(Zt_imag, Zt_real);
        return { magnitude: Math.log10(Zmag), phase: Zphase };
      });
    },
  },
];

export default function ImpedanceInput({ onSubmit, loading }: ImpedanceInputProps) {
  const [config, setConfig] = useState<Config | null>(null);
  const [magnitude, setMagnitude] = useState<number[]>([]);
  const [phase, setPhase] = useState<number[]>([]);
  const [selectedSample, setSelectedSample] = useState<number>(-1);
  const [manualInput, setManualInput] = useState('');
  const [inputMode, setInputMode] = useState<'sample' | 'manual'>('sample');

  useEffect(() => {
    getConfig()
      .then(setConfig)
      .catch((err) => console.error('Failed to load config:', err));
  }, []);

  const handleSampleSelect = (idx: number) => {
    if (!config) return;
    setSelectedSample(idx);

    const sample = SAMPLE_CURVES[idx];
    const data = sample.generate(config.frequencies);
    setMagnitude(data.map((d) => d.magnitude));
    setPhase(data.map((d) => d.phase));
  };

  const handleManualSubmit = () => {
    try {
      const data = JSON.parse(manualInput);
      if (!data.magnitude || !data.phase) {
        alert('JSON must have "magnitude" and "phase" arrays');
        return;
      }
      if (data.magnitude.length !== config?.num_freq || data.phase.length !== config?.num_freq) {
        alert(`Arrays must have ${config?.num_freq} elements`);
        return;
      }
      setMagnitude(data.magnitude);
      setPhase(data.phase);
    } catch {
      alert('Invalid JSON');
    }
  };

  const handleGenerate = () => {
    if (magnitude.length === 0 || phase.length === 0) {
      alert('Please select or input impedance data first');
      return;
    }
    onSubmit(magnitude, phase);
  };

  if (!config) {
    return (
      <div className="p-4 bg-secondary/50 rounded-lg animate-pulse flex flex-col gap-2">
        <div className="h-4 bg-muted/20 rounded w-3/4"></div>
        <div className="h-4 bg-muted/20 rounded w-1/2"></div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Mode Selection */}
      <div className="grid grid-cols-2 p-1 bg-secondary rounded-xl">
        <button
          onClick={() => setInputMode('sample')}
          className={`py-2 text-sm font-medium rounded-lg transition-all ${inputMode === 'sample'
              ? 'bg-white text-primary shadow-sm'
              : 'text-muted-foreground hover:text-foreground'
            }`}
        >
          Sample Curves
        </button>
        <button
          onClick={() => setInputMode('manual')}
          className={`py-2 text-sm font-medium rounded-lg transition-all ${inputMode === 'manual'
              ? 'bg-white text-primary shadow-sm'
              : 'text-muted-foreground hover:text-foreground'
            }`}
        >
          JSON Input
        </button>
      </div>

      {/* Sample Selection */}
      {inputMode === 'sample' && (
        <div className="space-y-3 animate-in fade-in duration-300">
          <div className="grid grid-cols-1 gap-2 max-h-[280px] overflow-y-auto pr-1 custom-scrollbar">
            {SAMPLE_CURVES.map((sample, idx) => (
              <button
                key={idx}
                onClick={() => handleSampleSelect(idx)}
                className={`p-3 text-left rounded-xl border transition-all hover:scale-[1.01] active:scale-[0.99] flex items-center gap-3 ${selectedSample === idx
                    ? 'border-primary bg-primary/10 ring-1 ring-primary/30 shadow-sm'
                    : 'border-transparent bg-secondary/50 hover:bg-secondary'
                  }`}
              >
                <span className="text-xl">{sample.icon}</span>
                <div className="flex-1 min-w-0">
                  <div className="font-bold text-foreground text-sm">{sample.name}</div>
                  <div className="text-[10px] text-muted-foreground truncate">{sample.desc}</div>
                </div>
                {selectedSample === idx && (
                  <div className="w-2 h-2 rounded-full bg-primary animate-pulse" />
                )}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Manual Input */}
      {inputMode === 'manual' && (
        <div className="space-y-3 animate-in fade-in duration-300">
          <textarea
            value={manualInput}
            onChange={(e) => setManualInput(e.target.value)}
            placeholder={`{\n  "magnitude": [...],\n  "phase": [...]\n}\n// details: ${config.num_freq} points`}
            className="w-full h-40 p-3 bg-secondary/30 border border-input rounded-xl font-mono text-xs focus:ring-2 focus:ring-primary focus:border-transparent outline-none resize-none"
          />
          <button
            onClick={handleManualSubmit}
            className="w-full py-2 bg-secondary hover:bg-secondary/80 text-foreground font-medium rounded-lg text-xs border border-border"
          >
            Parse JSON Data
          </button>
        </div>
      )}

      {/* Data Preview */}
      {magnitude.length > 0 && (
        <div className="p-3 bg-green-500/10 border border-green-500/20 rounded-xl flex justify-between items-center">
          <div>
            <p className="text-xs font-bold text-green-700 dark:text-green-400">
              Data Loaded Successfully
            </p>
            <p className="text-[10px] text-green-600/80 dark:text-green-500/80 font-mono mt-0.5">
              {magnitude.length} points • Range: [{Math.min(...magnitude).toFixed(1)}, {Math.max(...magnitude).toFixed(1)}]
            </p>
          </div>
          <div className="h-2 w-2 rounded-full bg-green-500 animate-pulse"></div>
        </div>
      )}

      {/* Generate Button */}
      <button
        onClick={handleGenerate}
        disabled={loading || magnitude.length === 0}
        className={`w-full py-3.5 rounded-xl font-bold text-sm uppercase tracking-wide transition-all transform shadow-lg ${loading || magnitude.length === 0
            ? 'bg-muted text-muted-foreground cursor-not-allowed shadow-none'
            : 'bg-primary text-primary-foreground hover:bg-primary/90 hover:-translate-y-0.5 shadow-primary/25'
          }`}
      >
        {loading ? (
          <span className="flex items-center justify-center gap-2">
            <span className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
            Generating...
          </span>
        ) : (
          'Generate Circuit'
        )}
      </button>

      {/* Info */}
      <div className="text-[10px] text-center text-muted-foreground font-mono">
        Range: {config.freq_min.toExponential(0)} - {config.freq_max.toExponential(0)} Hz
      </div>
    </div>
  );
}
