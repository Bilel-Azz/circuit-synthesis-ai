#!/bin/bash
# Monitor training and test model at each checkpoint

OVH_IP="57.128.57.31"
OVH_USER="ubuntu"

echo "🔍 Monitoring supervised training on OVH"
echo ""

while true; do
    # Check latest checkpoint
    LATEST=$(ssh "$OVH_USER@$OVH_IP" "ls -t ~/circuit_synthesis_gnn/outputs/gnn_supervised_v1/checkpoints/epoch_*.pt 2>/dev/null | head -1")

    if [ -z "$LATEST" ]; then
        echo "$(date '+%H:%M:%S') - No checkpoint yet, waiting..."
        sleep 60
        continue
    fi

    EPOCH=$(basename "$LATEST" | grep -oP '\d+')
    echo "$(date '+%H:%M:%S') - Latest checkpoint: epoch_${EPOCH}.pt"

    # Check if we already tested this epoch
    if [ -f "tested_epoch_${EPOCH}.flag" ]; then
        echo "  → Already tested epoch ${EPOCH}, waiting for next..."
        sleep 60
        continue
    fi

    echo "  → Testing epoch ${EPOCH}..."

    # Test with complex circuit
    ssh "$OVH_USER@$OVH_IP" "cd ~/circuit_synthesis_gnn && source ~/venv/bin/activate && python3 ~/test_complex_circuit.py" > "test_epoch_${EPOCH}.log" 2>&1

    # Download result
    scp "$OVH_USER@$OVH_IP":~/circuit_synthesis_gnn/test_complex_circuit.png "test_epoch_${EPOCH}.png"

    # Mark as tested
    touch "tested_epoch_${EPOCH}.flag"

    echo "  ✓ Test complete, saved to test_epoch_${EPOCH}.png"
    echo ""

    # Show summary
    grep "PREDICTED CIRCUIT:" -A 10 "test_epoch_${EPOCH}.log" | head -15

    # Check if training is done
    if [ "$EPOCH" -ge "50" ]; then
        echo ""
        echo "✅ Training complete (epoch 50 reached)"
        break
    fi

    # Wait before next check
    sleep 120
done

echo ""
echo "Monitoring stopped."
