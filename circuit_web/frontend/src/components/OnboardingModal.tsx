'use client';

import { useState, useEffect } from 'react';

interface OnboardingModalProps {
  onClose: () => void;
}

export default function OnboardingModal({ onClose }: OnboardingModalProps) {
  const [step, setStep] = useState(0);

  const steps = [
    {
      title: "Bienvenue sur Circuit Synthesis AI",
      icon: "🔬",
      content: (
        <div className="space-y-4">
          <p>
            Cet outil utilise l'<strong>intelligence artificielle</strong> pour generer
            automatiquement des circuits electriques RLC a partir d'une courbe d'impedance Z(f).
          </p>
          <div className="bg-primary/10 rounded-xl p-4 text-sm">
            <div className="font-bold mb-2">Comment ca marche ?</div>
            <div className="flex items-center gap-3">
              <div className="text-center">
                <div className="text-2xl mb-1">📈</div>
                <div className="text-xs">Courbe Z(f)</div>
              </div>
              <div className="text-xl">→</div>
              <div className="text-center">
                <div className="text-2xl mb-1">🤖</div>
                <div className="text-xs">Modele IA</div>
              </div>
              <div className="text-xl">→</div>
              <div className="text-center">
                <div className="text-2xl mb-1">⚡</div>
                <div className="text-xs">Circuit RLC</div>
              </div>
            </div>
          </div>
        </div>
      )
    },
    {
      title: "Comment utiliser l'outil",
      icon: "📝",
      content: (
        <div className="space-y-4">
          <div className="space-y-3">
            <div className="flex gap-3 items-start">
              <div className="w-6 h-6 rounded-full bg-primary text-white flex items-center justify-center text-sm font-bold shrink-0">1</div>
              <div>
                <div className="font-semibold">Choisir une courbe d'impedance</div>
                <div className="text-sm text-muted-foreground">Selectionnez un exemple pre-charge ou entrez vos propres donnees JSON</div>
              </div>
            </div>
            <div className="flex gap-3 items-start">
              <div className="w-6 h-6 rounded-full bg-primary text-white flex items-center justify-center text-sm font-bold shrink-0">2</div>
              <div>
                <div className="font-semibold">Generer le circuit</div>
                <div className="text-sm text-muted-foreground">Cliquez sur "Generer" et attendez quelques secondes</div>
              </div>
            </div>
            <div className="flex gap-3 items-start">
              <div className="w-6 h-6 rounded-full bg-primary text-white flex items-center justify-center text-sm font-bold shrink-0">3</div>
              <div>
                <div className="font-semibold">Analyser les resultats</div>
                <div className="text-sm text-muted-foreground">Visualisez le schema, comparez les courbes, exportez en SPICE</div>
              </div>
            </div>
          </div>
        </div>
      )
    },
    {
      title: "Synthese ≠ Reconnaissance",
      icon: "💡",
      content: (
        <div className="space-y-4">
          <div className="bg-yellow-500/10 border border-yellow-500/30 rounded-xl p-4">
            <div className="font-bold text-yellow-600 dark:text-yellow-400 mb-2">Point important !</div>
            <p className="text-sm">
              Le modele fait de la <strong>synthese</strong>, pas de la reconnaissance.
              Il ne devine pas le circuit original, mais genere un circuit <strong>fonctionnellement equivalent</strong>.
            </p>
          </div>

          <div className="bg-secondary/50 rounded-xl p-4 text-sm">
            <div className="font-semibold mb-2">Exemple :</div>
            <div className="grid grid-cols-3 gap-2 text-center">
              <div>
                <div className="font-mono text-xs bg-background rounded p-2">RL Serie</div>
                <div className="text-xs text-muted-foreground mt-1">Input</div>
              </div>
              <div className="flex items-center justify-center">
                <span className="text-lg">→ IA →</span>
              </div>
              <div>
                <div className="font-mono text-xs bg-background rounded p-2">R + 2L + C</div>
                <div className="text-xs text-muted-foreground mt-1">Output</div>
              </div>
            </div>
            <div className="text-center mt-3 text-xs text-muted-foreground">
              Les deux circuits produisent la <strong>meme courbe Z(f)</strong> !
            </div>
          </div>

          <p className="text-sm text-muted-foreground">
            C'est un <em>probleme inverse</em> : plusieurs circuits differents peuvent
            avoir la meme impedance. Ce qui compte, c'est le <strong>Score de Match</strong>
            entre la courbe cible et celle du circuit genere.
          </p>
        </div>
      )
    },
    {
      title: "Pret a commencer !",
      icon: "🚀",
      content: (
        <div className="space-y-4">
          <p>
            Vous pouvez maintenant utiliser l'outil. Quelques conseils :
          </p>
          <ul className="space-y-2 text-sm">
            <li className="flex gap-2">
              <span className="text-green-500">✓</span>
              <span>Commencez par les <strong>exemples pre-charges</strong> pour comprendre le fonctionnement</span>
            </li>
            <li className="flex gap-2">
              <span className="text-green-500">✓</span>
              <span>Un <strong>Match &gt; 80%</strong> indique une bonne correspondance</span>
            </li>
            <li className="flex gap-2">
              <span className="text-green-500">✓</span>
              <span>Utilisez l'<strong>export SPICE</strong> pour simuler dans LTspice</span>
            </li>
            <li className="flex gap-2">
              <span className="text-green-500">✓</span>
              <span>La topologie peut differer, seul le <strong>comportement</strong> compte</span>
            </li>
          </ul>

          <div className="bg-primary/10 rounded-xl p-4 text-center">
            <div className="text-sm text-muted-foreground mb-2">Projet de fin d'etudes</div>
            <div className="font-bold">Circuit Synthesis AI</div>
            <div className="text-xs text-muted-foreground mt-1">Transformer + Best-of-N Sampling</div>
          </div>
        </div>
      )
    }
  ];

  const currentStep = steps[step];
  const isLastStep = step === steps.length - 1;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
      {/* Backdrop */}
      <div
        className="absolute inset-0 bg-black/60 backdrop-blur-sm"
        onClick={onClose}
      />

      {/* Modal */}
      <div className="relative bg-background rounded-2xl shadow-2xl max-w-lg w-full max-h-[90vh] overflow-hidden border border-border">
        {/* Progress bar */}
        <div className="h-1 bg-secondary">
          <div
            className="h-full bg-primary transition-all duration-300"
            style={{ width: `${((step + 1) / steps.length) * 100}%` }}
          />
        </div>

        {/* Content */}
        <div className="p-6">
          {/* Header */}
          <div className="flex items-center gap-3 mb-4">
            <div className="text-3xl">{currentStep.icon}</div>
            <div>
              <h2 className="text-xl font-bold">{currentStep.title}</h2>
              <p className="text-xs text-muted-foreground">Etape {step + 1} sur {steps.length}</p>
            </div>
          </div>

          {/* Body */}
          <div className="text-foreground">
            {currentStep.content}
          </div>
        </div>

        {/* Footer */}
        <div className="flex items-center justify-between p-4 border-t border-border bg-secondary/30">
          <button
            onClick={onClose}
            className="text-sm text-muted-foreground hover:text-foreground transition-colors"
          >
            Passer
          </button>

          <div className="flex gap-2">
            {step > 0 && (
              <button
                onClick={() => setStep(step - 1)}
                className="px-4 py-2 text-sm font-medium rounded-lg border border-border hover:bg-secondary transition-colors"
              >
                Precedent
              </button>
            )}
            <button
              onClick={() => {
                if (isLastStep) {
                  onClose();
                } else {
                  setStep(step + 1);
                }
              }}
              className="px-4 py-2 text-sm font-medium rounded-lg bg-primary text-primary-foreground hover:bg-primary/90 transition-colors"
            >
              {isLastStep ? "Commencer" : "Suivant"}
            </button>
          </div>
        </div>

        {/* Step indicators */}
        <div className="absolute bottom-20 left-1/2 -translate-x-1/2 flex gap-1.5">
          {steps.map((_, i) => (
            <button
              key={i}
              onClick={() => setStep(i)}
              className={`w-2 h-2 rounded-full transition-colors ${
                i === step ? 'bg-primary' : 'bg-border hover:bg-muted-foreground'
              }`}
            />
          ))}
        </div>
      </div>
    </div>
  );
}
