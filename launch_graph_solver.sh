#!/bin/bash
# Launch GraphSolver training on OVH (more stable than RobustSolver)

OVH_IP="57.128.57.31"
OVH_USER="ubuntu"

echo "🚀 Lancement training GraphSolver sur OVH"
echo ""

ssh "$OVH_USER@$OVH_IP" << 'EOF'
# Arrêter ancien training
pkill -9 python
sleep 2

cd ~/circuit_synthesis_gnn
source ~/venv/bin/activate

echo "=== Lancement GraphSolver Training ==="
nohup python scripts/train_graph_solver.py \
    --data outputs/data/gnn_750k_rlc.pt \
    --epochs 50 \
    --lr 0.0003 \
    --batch-size 128 \
    --sparsity-weight 0.3 \
    --connectivity-weight 0.2 \
    --tau-end 0.3 \
    --tau-anneal-epochs 50 \
    --output-dir outputs/gnn_graph_solver_v1 \
    --save-every 5 \
    --no-refinement \
    > training_graph_solver.log 2>&1 &

echo "Training lancé! PID: $!"
echo ""
echo "Monitoring:"
echo "  tail -f ~/circuit_synthesis_gnn/training_graph_solver.log"
sleep 10

echo ""
echo "=== Premières lignes log ==="
tail -30 ~/circuit_synthesis_gnn/training_graph_solver.log
EOF

echo ""
echo "✅ Training GraphSolver lancé!"
echo ""
echo "Pour surveiller:"
echo "  ssh $OVH_USER@$OVH_IP"
echo "  tail -f ~/circuit_synthesis_gnn/training_graph_solver.log"
