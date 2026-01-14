#!/bin/bash
# fix_nan_ovh.sh - Fix rapide pour gérer les NaN dans les métriques

OVH_IP="57.128.57.31"
OVH_USER="ubuntu"
SSH_KEY="$HOME/.ssh/ovh_rsa"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${YELLOW}🔧 Fix NaN sur OVH${NC}"
echo ""

ssh -i "$SSH_KEY" "$OVH_USER@$OVH_IP" << 'EOF'
cd ~/circuit_synthesis_gnn

echo "=== Backup loss.py ==="
cp training/loss.py training/loss.py.bak

echo "=== Modification loss.py pour gérer NaN ==="
cat > /tmp/fix_nan.py << 'PYFIX'
import sys

# Lire le fichier
with open('training/loss.py', 'r') as f:
    lines = f.readlines()

# Trouver et remplacer la section compute_impedance_loss
new_lines = []
in_impedance_loss = False
replaced = False

for i, line in enumerate(lines):
    if 'def compute_impedance_loss(' in line:
        in_impedance_loss = True

    # Trouver la ligne "# Compute percentage errors"
    if in_impedance_loss and '# Compute percentage errors' in line:
        # Ajouter notre fix
        new_lines.append(line)
        new_lines.append('    # Protection contre NaN\n')
        new_lines.append('    if torch.isnan(mag_loss) or torch.isnan(phase_loss):\n')
        new_lines.append('        print(f"WARNING: NaN detected in loss! mag_loss={mag_loss}, phase_loss={phase_loss}")\n')
        new_lines.append('        mag_error_pct = 999.9\n')
        new_lines.append('        phase_error_pct = 999.9\n')
        new_lines.append('        combined_error = 999.9\n')
        new_lines.append('    else:\n')
        # Indenter les lignes suivantes
        next_line_idx = i + 1
        while next_line_idx < len(lines) and not lines[next_line_idx].strip().startswith('metrics = {'):
            # Indenter avec 4 espaces supplémentaires
            new_lines.append('    ' + lines[next_line_idx])
            next_line_idx += 1
        # Ajouter la ligne metrics
        new_lines.append(lines[next_line_idx])

        # Skip les lignes déjà traitées
        for j in range(i + 1, next_line_idx + 1):
            lines[j] = None  # Marquer comme déjà traité

        replaced = True
        continue

    # Skip les lignes déjà traitées
    if line is None:
        continue

    new_lines.append(line)

    if in_impedance_loss and line.strip().startswith('return total_loss'):
        in_impedance_loss = False

# Écrire le fichier modifié
with open('training/loss.py', 'w') as f:
    f.writelines(new_lines)

if replaced:
    print("✓ loss.py modifié avec succès")
else:
    print("✗ Échec de la modification")
    sys.exit(1)
PYFIX

python3 /tmp/fix_nan.py

if [ $? -eq 0 ]; then
    echo ""
    echo "=== Vérification modification ==="
    grep -A 10 "Protection contre NaN" training/loss.py

    echo ""
    echo "✅ Fix appliqué!"
else
    echo ""
    echo "❌ Erreur lors du fix, restauration backup..."
    cp training/loss.py.bak training/loss.py
fi
EOF

echo ""
echo -e "${GREEN}✅ Fix terminé!${NC}"
echo ""
echo "Prochaine étape: Relancer le training sur OVH"
echo "  ssh -i $SSH_KEY $OVH_USER@$OVH_IP"
echo "  pkill -9 python  # Arrêter ancien training"
echo "  cd ~/circuit_synthesis_gnn"
echo "  source ~/venv/bin/activate"
echo "  python scripts/train.py ... 2>&1 | tee training.log"
