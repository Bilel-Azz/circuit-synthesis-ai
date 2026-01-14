'use client';

import { useState, useEffect } from 'react';
import ImpedanceInput from '@/components/ImpedanceInput';
import ImpedanceChart from '@/components/ImpedanceChart';
import CircuitDisplay from '@/components/CircuitDisplay';
import OnboardingModal from '@/components/OnboardingModal';
import { generateCircuit, checkHealth, GenerateResponse, Impedance, HealthCheck } from '@/lib/api';

export default function Home() {
  const [health, setHealth] = useState<HealthCheck | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<GenerateResponse | null>(null);
  const [targetImpedance, setTargetImpedance] = useState<Impedance | null>(null);
  const [numCandidates, setNumCandidates] = useState(100);
  const [tau, setTau] = useState(0.5);
  const [showOnboarding, setShowOnboarding] = useState(false);

  // Show onboarding on first visit
  useEffect(() => {
    const hasSeenOnboarding = localStorage.getItem('circuit-ai-onboarding-seen');
    if (!hasSeenOnboarding) {
      setShowOnboarding(true);
    }
  }, []);

  const handleCloseOnboarding = () => {
    setShowOnboarding(false);
    localStorage.setItem('circuit-ai-onboarding-seen', 'true');
  };

  // Check backend health on load
  useEffect(() => {
    checkHealth()
      .then(setHealth)
      .catch(() => setHealth(null));
  }, []);

  const handleGenerate = async (magnitude: number[], phase: number[]) => {
    setLoading(true);
    setError(null);
    setResult(null);

    // Store target impedance for display
    setTargetImpedance({
      magnitude,
      phase,
      frequencies: [], // Will be filled from result
    });

    try {
      const response = await generateCircuit(magnitude, phase, {
        tau,
        num_candidates: numCandidates,
      });

      if (response.success && response.best) {
        // Update target with frequencies from response
        setTargetImpedance({
          magnitude,
          phase,
          frequencies: response.best.impedance.frequencies,
        });
      }

      setResult(response);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="container mx-auto px-4 py-8 max-w-7xl min-h-screen flex flex-col">
      {/* Onboarding Modal */}
      {showOnboarding && <OnboardingModal onClose={handleCloseOnboarding} />}

      {/* Header */}
      <header className="flex flex-col md:flex-row md:items-center justify-between mb-12 gap-6">
        <div>
          <h1 className="text-4xl md:text-5xl font-extrabold tracking-tight text-primary bg-clip-text">
            Circuit Synthesis <span className="text-accent">AI</span>
          </h1>
          <p className="text-muted-foreground mt-2 text-lg">
            Generate electrical equivalence from impedance curves using Transformer models.
          </p>
        </div>

        {/* Status Badge + Help Button */}
        <div className="flex items-center gap-3 flex-shrink-0">
          <button
            onClick={() => setShowOnboarding(true)}
            className="p-2 rounded-full hover:bg-secondary transition-colors"
            title="Aide"
          >
            <svg className="w-5 h-5 text-muted-foreground" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8.228 9c.549-1.165 2.03-2 3.772-2 2.21 0 4 1.343 4 3 0 1.4-1.278 2.575-3.006 2.907-.542.104-.994.54-.994 1.093m0 3h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
          </button>
          {health ? (
            <div className="glass-card px-4 py-2 rounded-full flex items-center gap-2 border-green-200/50 bg-green-50/50 text-green-700">
              <span className="relative flex h-3 w-3">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-green-400 opacity-75"></span>
                <span className="relative inline-flex rounded-full h-3 w-3 bg-green-500"></span>
              </span>
              <span className="font-medium text-sm">System Online ({health.device})</span>
            </div>
          ) : (
            <div className="glass-card px-4 py-2 rounded-full flex items-center gap-2 border-red-200/50 bg-red-50/50 text-red-700">
              <span className="h-3 w-3 rounded-full bg-red-500"></span>
              <span className="font-medium text-sm">Backend Disconnected</span>
            </div>
          )}
        </div>
      </header>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 flex-grow">
        {/* Left Panel: Controls */}
        <div className="lg:col-span-4 space-y-6">
          <section className="glass-card rounded-2xl p-6 relative overflow-hidden">
            <div className="absolute top-0 right-0 w-32 h-32 bg-primary/5 rounded-full blur-3xl -z-10" />

            <h2 className="text-xl font-bold mb-6 flex items-center gap-2">
              <svg className="w-5 h-5 text-primary" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
              </svg>
              Input Configuration
            </h2>
            <ImpedanceInput onSubmit={handleGenerate} loading={loading} />
          </section>

          <section className="glass-card rounded-2xl p-6">
            <h2 className="text-xl font-bold mb-6 flex items-center gap-2">
              <svg className="w-5 h-5 text-primary" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6V4m0 2a2 2 0 100 4m0-4a2 2 0 110 4m-6 8a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4m6 6v10m6-2a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4" />
              </svg>
              Model Parameters
            </h2>

            <div className="space-y-6">
              <div className="space-y-3">
                <div className="flex justify-between">
                  <label className="text-sm font-medium text-muted-foreground">Candidates (Best-of-N)</label>
                  <span className="text-sm font-mono font-bold text-primary bg-primary/10 px-2 py-0.5 rounded">{numCandidates}</span>
                </div>
                <input
                  type="range"
                  min="10"
                  max="200"
                  step="10"
                  value={numCandidates}
                  onChange={(e) => setNumCandidates(Number(e.target.value))}
                  className="w-full h-2 bg-secondary rounded-lg appearance-none cursor-pointer accent-primary"
                />
              </div>

              <div className="space-y-3">
                <div className="flex justify-between">
                  <label className="text-sm font-medium text-muted-foreground">Temperature (τ)</label>
                  <span className="text-sm font-mono font-bold text-primary bg-primary/10 px-2 py-0.5 rounded">{tau.toFixed(1)}</span>
                </div>
                <input
                  type="range"
                  min="0.1"
                  max="2"
                  step="0.1"
                  value={tau}
                  onChange={(e) => setTau(Number(e.target.value))}
                  className="w-full h-2 bg-secondary rounded-lg appearance-none cursor-pointer accent-primary"
                />
              </div>
            </div>
          </section>
        </div>

        {/* Right Panel: Data & Results */}
        <div className="lg:col-span-8 space-y-6">
          {error && (
            <div className="glass-card border-red-200 bg-red-50/50 p-6 rounded-2xl flex items-start gap-4">
              <div className="p-2 bg-red-100 rounded-lg text-red-600">
                <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
              </div>
              <div>
                <h3 className="text-lg font-bold text-red-800">Generation Failed</h3>
                <p className="text-red-700 mt-1">{error}</p>
              </div>
            </div>
          )}

          {loading && (
            <div className="glass-card p-12 rounded-2xl flex flex-col items-center justify-center min-h-[400px]">
              <div className="relative w-20 h-20 mb-8">
                <div className="absolute inset-0 border-4 border-primary/20 rounded-full"></div>
                <div className="absolute inset-0 border-4 border-primary border-t-transparent rounded-full animate-spin"></div>
              </div>
              <h3 className="text-2xl font-bold text-foreground">Synthesizing Circuit</h3>
              <p className="text-muted-foreground mt-2">Running inference on {numCandidates} candidates...</p>
            </div>
          )}

          {!loading && !result && !error && (
            <div className="glass-card p-12 rounded-2xl flex flex-col items-center justify-center min-h-[500px] text-center">
              <div className="w-32 h-32 bg-primary/5 rounded-full flex items-center justify-center mb-6">
                <svg className="w-16 h-16 text-primary/40" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 3v2m6-2v2M9 19v2m6-2v2M5 9H3m2 6H3m18-6h-2m2 6h-2M7 19h10a2 2 0 002-2V7a2 2 0 00-2-2H7a2 2 0 00-2 2v10a2 2 0 002 2zM9 9h6v6H9V9z" />
                </svg>
              </div>
              <h2 className="text-2xl font-bold text-foreground mb-3">Ready to Generate</h2>
              <p className="text-muted-foreground max-w-md mx-auto">
                Configure your impedance targets on the left panel and click "Generate Circuit" to start the AI synthesis process.
              </p>
            </div>
          )}

          {result && result.success && result.best && (
            <div className="space-y-6 animate-in fade-in slide-in-from-bottom-4 duration-500">
              {/* Key Metrics */}
              <div className="grid grid-cols-3 gap-4">
                <div className="glass-card p-6 rounded-2xl text-center">
                  <p className="text-3xl font-bold text-primary font-mono">{result.best.error.magnitude.toFixed(3)}</p>
                  <p className="text-sm font-medium text-muted-foreground mt-1">Magnitude Error (RMSE)</p>
                </div>
                <div className="glass-card p-6 rounded-2xl text-center">
                  <p className="text-3xl font-bold text-accent font-mono">{(result.best.error.phase * 180 / Math.PI).toFixed(1)}°</p>
                  <p className="text-sm font-medium text-muted-foreground mt-1">Phase Error (RMSE)</p>
                </div>
                <div className="glass-card p-6 rounded-2xl text-center">
                  <p className="text-3xl font-bold text-foreground font-mono">{result.num_candidates}</p>
                  <p className="text-sm font-medium text-muted-foreground mt-1">Candidates Evaluated</p>
                </div>
              </div>

              {/* Generation Stats */}
              {result.stats && (
                <div className="glass-card p-4 rounded-2xl">
                  <div className="flex items-center justify-between text-sm">
                    <span className="text-muted-foreground">Generation Stats:</span>
                    <div className="flex gap-4">
                      <span className="text-green-600 font-medium">
                        ✓ {result.stats.valid} valid
                      </span>
                      <span className="text-red-500 font-medium">
                        ✗ {result.stats.invalid} invalid
                      </span>
                      {result.stats.empty > 0 && (
                        <span className="text-yellow-600 font-medium">
                          ○ {result.stats.empty} empty
                        </span>
                      )}
                      <span className="text-muted-foreground">
                        / {result.stats.total} total
                      </span>
                    </div>
                  </div>
                </div>
              )}

              {/* Charts */}
              <div className="glass-card p-6 rounded-2xl">
                <ImpedanceChart
                  target={targetImpedance || undefined}
                  predicted={result.best.impedance}
                  title="Impedance Response"
                />
              </div>

              {/* Circuit Diagram */}
              <div className="glass-card p-6 rounded-2xl">
                <CircuitDisplay
                  components={result.best.components}
                  title="Synthesized Circuit Topology"
                />
              </div>

              {/* Candidates List */}
              {result.candidates.length > 1 && (
                <div className="glass-card p-6 rounded-2xl">
                  <h3 className="text-lg font-bold mb-4">Alternative Candidates</h3>
                  <div className="space-y-2 max-h-60 overflow-y-auto pr-2 custom-scrollbar">
                    {result.candidates.slice(1, 6).map((cand, idx) => (
                      <div
                        key={idx}
                        className="p-3 bg-secondary/30 rounded-lg flex items-center justify-between hover:bg-secondary/50 transition-colors"
                      >
                        <div className="flex gap-2">
                          {cand.components.map((c, i) => (
                            <span
                              key={i}
                              className={`px-2 py-0.5 rounded text-xs font-mono font-medium ${c.type === 'R'
                                  ? 'bg-orange-100 text-orange-700'
                                  : c.type === 'L'
                                    ? 'bg-blue-100 text-blue-700'
                                    : 'bg-green-100 text-green-700'
                                }`}
                            >
                              {c.type}={c.formatted_value}
                            </span>
                          ))}
                        </div>
                        <span className="text-sm text-muted-foreground font-mono">
                          Err: {cand.error.total.toFixed(4)}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </main>
  );
}
