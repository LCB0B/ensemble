#!/usr/bin/env bash
# Benchmark generation throughput for different merged batch sizes.
# Collects gen_toks/s and total_toks/s from the script output.
#
# Usage:
#   chmod +x scripts/benchmark_generation.sh
#   ./scripts/benchmark_generation.sh
#
# Optional environment overrides:
#   MODEL=checkpoints/transformer/destiny/model/best.ckpt
#   HPARAMS=configs/destiny/hparams_destiny_pretrain.yaml
#   PROMPT_LEN=64
#   MAX_NEW=512
#   SIZES="32 64 96 128 160"
#   NUM_BATCHES=1
#   TEMP=1.0
#   TOP_P=0.95
#   EXTRA="--compile_model --use_cache"

set -euo pipefail

MODEL="${MODEL:-checkpoints/transformer/destiny/model/best.ckpt}"
HPARAMS="${HPARAMS:-configs/destiny/hparams_destiny_pretrain.yaml}"
PROMPT_LEN="${PROMPT_LEN:-64}"
MAX_NEW="${MAX_NEW:-256}"
SIZES="${SIZES:-32 64 96 128}"
NUM_BATCHES="${NUM_BATCHES:-1}"
TEMP="${TEMP:-1.0}"
TOP_P="${TOP_P:-}"
EXTRA="${EXTRA:-}"
LOG_DIR="benchmark_logs"
mkdir -p "$LOG_DIR"

echo "# Benchmark config"
echo "# MODEL=$MODEL"
echo "# HPARAMS=$HPARAMS"
echo "# PROMPT_LEN=$PROMPT_LEN  MAX_NEW=$MAX_NEW  NUM_BATCHES=$NUM_BATCHES  TEMP=$TEMP  TOP_P=${TOP_P:-None}"
echo "# EXTRA args: $EXTRA"
echo

# Warm-up (small run to trigger any compile / lazy init)
echo "Warm-up run..."
python scripts/generate_sequences.py \
  --model_path "$MODEL" \
  --hparams_path "$HPARAMS" \
  --prompt_length "$PROMPT_LEN" \
  --max_new_tokens "$MAX_NEW" \
  --num_batches 1 \
  --target_batch_size 8 \
  --temperature "$TEMP" \
  ${TOP_P:+--top_p "$TOP_P"} \
  --profile $EXTRA > /dev/null 2>&1 || true
echo "Warm-up done."
echo

printf "%-10s %-12s %-12s %-10s %-10s\n" "BATCH" "TOT_TOKS/S" "GEN_TOKS/S" "WALL_MS" "FIRST_MS"
printf "%-10s %-12s %-12s %-10s %-10s\n" "-----" "----------" "----------" "-------" "-------"

for B in $SIZES; do
  log="$LOG_DIR/run_bs${B}.log"
  echo "Running target_batch_size=$B ..."
  set +e
  out=$(python scripts/generate_sequences.py \
      --model_path "$MODEL" \
      --hparams_path "$HPARAMS" \
      --prompt_length "$PROMPT_LEN" \
      --max_new_tokens "$MAX_NEW" \
      --num_batches "$NUM_BATCHES" \
      --target_batch_size "$B" \
      --temperature "$TEMP" \
      ${TOP_P:+--top_p "$TOP_P"} \
      --profile $EXTRA 2>&1)
  status=$?
  set -e
  echo "$out" > "$log"

  # Prefer aggregated line; fallback to last Metrics line with tokens_per_sec_total
  agg_line=$(echo "$out" | grep '\[AGG SPEED\]' | tail -1 || true)
  metrics_line=$(echo "$out" | grep "Metrics:" | tail -1 || true)

  if [[ -n "$agg_line" ]]; then
    # Example: [AGG SPEED] total_prompt_tokens=1204 total_gen_tokens=20000 wall_ms=8705.5 total_toks/s=2435.7 gen_toks/s=2297.4
    wall_ms=$(echo "$agg_line" | sed -n 's/.*wall_ms=\([0-9.]*\).*/\1/p')
    total_tps=$(echo "$agg_line" | sed -n 's/.*total_toks\/s=\([0-9.]*\).*/\1/p')
    gen_tps=$(echo "$agg_line" | sed -n 's/.*gen_toks\/s=\([0-9.]*\).*/\1/p')
    first_ms="-"  # Not present on agg line
  else
    # Parse from metrics JSON-like dict string
    # Metrics: {'... 'tokens_per_sec_total': '2435.70', 'gen_tokens_per_sec': '2297.40', 'first_step_ms': '1236.25'}
    total_tps=$(echo "$metrics_line" | sed -n "s/.*'tokens_per_sec_total': '\([0-9.]*\)'.*/\1/p")
    gen_tps=$(echo "$metrics_line" | sed -n "s/.*'gen_tokens_per_sec': '\([0-9.]*\)'.*/\1/p")
    first_ms=$(echo "$metrics_line" | sed -n "s/.*'first_step_ms': '\([0-9.]*\)'.*/\1/p")
    wall_ms="-"
  fi

  # If any missing, mark as NA
  [[ -z "$total_tps" ]] && total_tps="NA"
  [[ -z "$gen_tps" ]] && gen_tps="NA"
  [[ -z "$wall_ms" ]] && wall_ms="NA"
  [[ -z "$first_ms" ]] && first_ms="NA"

  printf "%-10s %-12s %-12s %-10s %-10s\n" "$B" "$total_tps" "$gen_tps" "$wall_ms" "$first_ms"

  if [[ $status -ne 0 ]]; then
    echo "  (Run returned non-zero exit $status, see $log)"
  fi
done

echo
echo "Logs saved under $LOG_DIR/"