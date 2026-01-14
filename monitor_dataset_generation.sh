#!/bin/bash
# Monitor dataset generation on OVH

OVH_IP="57.128.57.31"
OVH_USER="ubuntu"

echo "📊 Monitoring dataset generation on OVH..."
echo "Press Ctrl+C to exit monitoring"
echo ""

ssh "$OVH_USER@$OVH_IP" "tail -f ~/circuit_synthesis_gnn/dataset_generation.log"
