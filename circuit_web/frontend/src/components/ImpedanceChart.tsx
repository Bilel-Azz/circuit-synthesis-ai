'use client';

import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';
import { Impedance } from '@/lib/api';

interface ImpedanceChartProps {
  target?: Impedance;
  predicted?: Impedance;
  title?: string;
}

const CustomTooltip = ({ active, payload, label, unit }: any) => {
  if (active && payload && payload.length) {
    return (
      <div className="glass-card p-3 rounded-lg border border-border bg-background/95 shadow-xl text-xs">
        <p className="font-bold text-foreground mb-2">{Number(label).toFixed(1)} Hz</p>
        {payload.map((entry: any, index: number) => (
          <div key={index} className="flex items-center gap-2 mb-1 last:mb-0">
            <span
              className="w-2 h-2 rounded-full"
              style={{ backgroundColor: entry.color }}
            />
            <span className="text-muted-foreground">{entry.name}:</span>
            <span className="font-mono font-medium text-foreground">
              {entry.value.toFixed(3)} {unit}
            </span>
          </div>
        ))}
      </div>
    );
  }
  return null;
};

export default function ImpedanceChart({ target, predicted, title }: ImpedanceChartProps) {
  if (!target && !predicted) {
    return (
      <div className="w-full h-64 flex flex-col items-center justify-center bg-secondary/20 rounded-xl border border-dashed border-border text-muted-foreground">
        <svg className="w-10 h-10 mb-2 opacity-20" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 12l3-3 3 3 4-4M8 21l4-4 4 4M3 4h18M4 4h16v12a1 1 0 01-1 1H5a1 1 0 01-1-1V4z" />
        </svg>
        <p className="text-sm font-medium">No impedance data available</p>
      </div>
    );
  }

  const frequencies = target?.frequencies || predicted?.frequencies || [];

  // Calculate correlation/match score if both present
  let matchScore = null;
  if (target && predicted && target.magnitude.length === predicted.magnitude.length) {
    const n = target.magnitude.length;
    let magError = 0;
    let phaseError = 0;
    for (let i = 0; i < n; i++) {
      magError += Math.pow(target.magnitude[i] - predicted.magnitude[i], 2);
      phaseError += Math.pow(target.phase[i] - predicted.phase[i], 2);
    }
    const magRMSE = Math.sqrt(magError / n);
    const phaseRMSE = Math.sqrt(phaseError / n);
    // Convert to a 0-100% score (assuming max reasonable error of ~2 for magnitude, ~1rad for phase)
    const magScore = Math.max(0, 100 * (1 - magRMSE / 2));
    const phaseScore = Math.max(0, 100 * (1 - phaseRMSE / 1));
    matchScore = Math.round((magScore + phaseScore) / 2);
  }

  // Prepare data for magnitude chart
  const magnitudeData = frequencies.map((freq, i) => ({
    frequency: freq,
    target: target?.magnitude[i],
    predicted: predicted?.magnitude[i],
  }));

  // Prepare data for phase chart (convert to degrees)
  const phaseData = frequencies.map((freq, i) => ({
    frequency: freq,
    target: target ? target.phase[i] * 180 / Math.PI : undefined,
    predicted: predicted ? predicted.phase[i] * 180 / Math.PI : undefined,
  }));

  return (
    <div className="space-y-6">
      {title && (
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="w-1 h-5 bg-primary rounded-full"></div>
            <h3 className="text-lg font-bold text-foreground">{title}</h3>
          </div>
          {matchScore !== null && (
            <div className={`flex items-center gap-2 px-3 py-1.5 rounded-full text-sm font-bold ${
              matchScore >= 80 ? 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400' :
              matchScore >= 60 ? 'bg-yellow-100 text-yellow-700 dark:bg-yellow-900/30 dark:text-yellow-400' :
              'bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400'
            }`}>
              <span className={`w-2 h-2 rounded-full ${
                matchScore >= 80 ? 'bg-green-500' :
                matchScore >= 60 ? 'bg-yellow-500' : 'bg-red-500'
              }`}></span>
              Match: {matchScore}%
            </div>
          )}
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Magnitude Chart */}
        <div className="bg-white/50 dark:bg-black/20 p-4 rounded-2xl border border-border">
          <h4 className="text-xs font-bold uppercase tracking-wider text-muted-foreground mb-4 pl-2">Magnitude (log10|Z|)</h4>
          <div className="w-full h-[300px]">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={magnitudeData} margin={{ top: 5, right: 20, bottom: 20, left: 0 }}>
                <CartesianGrid strokeDasharray="3 3" opacity={0.3} vertical={false} />
                <XAxis
                  dataKey="frequency"
                  scale="log"
                  domain={['dataMin', 'dataMax']}
                  tickFormatter={(v) => v >= 1000 ? `${v / 1000}k` : v.toFixed(0)}
                  tick={{ fontSize: 10, fill: 'hsl(var(--muted-foreground))' }}
                  axisLine={false}
                  tickLine={false}
                  dy={10}
                />
                <YAxis
                  tick={{ fontSize: 10, fill: 'hsl(var(--muted-foreground))' }}
                  axisLine={false}
                  tickLine={false}
                  width={40}
                />
                <Tooltip content={<CustomTooltip unit="" />} />
                <Legend iconType="circle" wrapperStyle={{ fontSize: '12px', paddingTop: '10px' }} />
                {target && (
                  <Line
                    type="monotone"
                    dataKey="target"
                    stroke="hsl(var(--primary))"
                    strokeWidth={2.5}
                    dot={false}
                    name="Cible (Input)"
                    activeDot={{ r: 4, strokeWidth: 0 }}
                  />
                )}
                {predicted && (
                  <Line
                    type="monotone"
                    dataKey="predicted"
                    stroke="#f97316"
                    strokeWidth={2}
                    strokeDasharray="5 5"
                    dot={false}
                    name="Circuit genere"
                    activeDot={{ r: 4, strokeWidth: 0 }}
                  />
                )}
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Phase Chart */}
        <div className="bg-white/50 dark:bg-black/20 p-4 rounded-2xl border border-border">
          <h4 className="text-xs font-bold uppercase tracking-wider text-muted-foreground mb-4 pl-2">Phase (degres)</h4>
          <div className="w-full h-[300px]">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={phaseData} margin={{ top: 5, right: 20, bottom: 20, left: 0 }}>
                <CartesianGrid strokeDasharray="3 3" opacity={0.3} vertical={false} />
                <XAxis
                  dataKey="frequency"
                  scale="log"
                  domain={['dataMin', 'dataMax']}
                  tickFormatter={(v) => v >= 1000 ? `${v / 1000}k` : v.toFixed(0)}
                  tick={{ fontSize: 10, fill: 'hsl(var(--muted-foreground))' }}
                  axisLine={false}
                  tickLine={false}
                  dy={10}
                />
                <YAxis
                  domain={[-90, 90]}
                  tick={{ fontSize: 10, fill: 'hsl(var(--muted-foreground))' }}
                  axisLine={false}
                  tickLine={false}
                  width={40}
                />
                <Tooltip content={<CustomTooltip unit="°" />} />
                <Legend iconType="circle" wrapperStyle={{ fontSize: '12px', paddingTop: '10px' }} />
                {target && (
                  <Line
                    type="monotone"
                    dataKey="target"
                    stroke="hsl(var(--primary))"
                    strokeWidth={2.5}
                    dot={false}
                    name="Cible (Input)"
                    activeDot={{ r: 4, strokeWidth: 0 }}
                  />
                )}
                {predicted && (
                  <Line
                    type="monotone"
                    dataKey="predicted"
                    stroke="#f97316"
                    strokeWidth={2}
                    strokeDasharray="5 5"
                    dot={false}
                    name="Circuit genere"
                    activeDot={{ r: 4, strokeWidth: 0 }}
                  />
                )}
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Comparison Legend */}
      {target && predicted && (
        <div className="flex items-center justify-center gap-6 text-xs text-muted-foreground">
          <div className="flex items-center gap-2">
            <div className="w-6 h-0.5 bg-primary"></div>
            <span>Courbe cible (entree)</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-6 h-0.5 bg-orange-500" style={{ backgroundImage: 'repeating-linear-gradient(90deg, #f97316 0, #f97316 5px, transparent 5px, transparent 10px)' }}></div>
            <span>Impedance du circuit genere</span>
          </div>
        </div>
      )}
    </div>
  );
}
