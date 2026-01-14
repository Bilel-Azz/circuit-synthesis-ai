#!/bin/bash
# Launch SUPERVISED training on OVH (stable, no solver during training)

OVH_IP="57.128.57.31"
OVH_USER="ubuntu"

echo "🚀 Lancement training SUPERVISED sur OVH"
echo ""

ssh "$OVH_USER@$OVH_IP" << 'EOF'
# Arrêter ancien training
pkill -9 python
sleep 2

cd ~/circuit_synthesis_gnn
source ~/venv/bin/activate

echo "=== Lancement Supervised Training ==="
nohup python scripts/train_supervised.py \
    --data outputs/data/gnn_750k_rlc.pt \
    --epochs 50 \
    --lr 0.0003 \
    --batch-size 128 \
    --type-weight 1.0 \
    --value-weight 1.0 \
    --nodes-weight 0.5 \
    --tau-end 0.3 \
    --tau-anneal-epochs 50 \
    --output-dir outputs/gnn_supervised_rlc_v1 \
    --save-every 5 \
    > training_supervised.log 2>&1 &

echo "Training lancé! PID: $!"
echo ""
echo "Monitoring:"
echo "  tail -f ~/circuit_synthesis_gnn/training_supervised.log"
sleep 10

echo ""
echo "=== Premières lignes log ==="
tail -30 ~/circuit_synthesis_gnn/training_supervised.log
EOF

echo ""
echo "✅ Training SUPERVISED lancé!"
echo ""
echo "Pour surveiller:"
echo "  ssh $OVH_USER@$OVH_IP"
echo "  tail -f ~/circuit_synthesis_gnn/training_supervised.log"
