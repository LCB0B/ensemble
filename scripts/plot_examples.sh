#!/bin/bash
# Example usage of the training loss plotting script

echo "📊 Training Loss Plotting Examples"
echo "=================================="

echo ""
echo "1. List all available experiments:"
echo "python scripts/plot_training_loss.py --list"

echo ""
echo "2. Show available metrics for a specific experiment:"
echo "python scripts/plot_training_loss.py 063_fearless_cheetah --show-metrics"

echo ""
echo "3. Plot with auto-detected metrics:"
echo "python scripts/plot_training_loss.py 063_fearless_cheetah"

echo ""
echo "4. Plot specific metrics:"
echo "python scripts/plot_training_loss.py 063_fearless_cheetah --metrics val/loss \"MLM top1\""

echo ""
echo "5. Compare multiple experiments:"
echo "python scripts/plot_training_loss.py --compare 061 062 063 --metric val/loss"

echo ""
echo "6. Save plot to file:"
echo "python scripts/plot_training_loss.py 063_fearless_cheetah --save my_training_plot.png"

echo ""
echo "7. Filter experiments by pattern:"
echo "python scripts/plot_training_loss.py --list --pattern fearless"

echo ""
echo "🚀 Try these commands to explore your training logs!"