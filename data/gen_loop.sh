#!/bin/zsh
# Keep gen_stories.py running until the target is reached. The generator is resumable,
# so a crash (Metal OOM after many hours) just costs the current batch.
# A battery guard pauses generation (PAUSE file) when a weak adapter cannot keep up
# with the GPU load, and resumes once the battery has recharged.
ROOT=/Volumes/Modelltest/pytte
PY=$ROOT/.venv/bin/python
export HF_HOME=$ROOT/hf-cache
PAUSE_LOW=30     # pause below this battery %
PAUSE_HIGH=90    # resume at or above this %
BIG_ADAPTER=60   # watts; at or above this the guard never pauses
GEN_DIR=${GEN_DIR:-$ROOT/data/sagosvenska}   # PAUSE/STOP live here; set GEN_DIR for v2 runs
cd $GEN_DIR

battery_guard() {
  while true; do
    pct=$(pmset -g batt | grep -o '[0-9]*%' | tr -d %)
    watts=$(pmset -g ac 2>/dev/null | awk '/Wattage/ {print $3+0}')
    ac=$(pmset -g batt | grep -c "AC Power")
    if [[ -f PAUSE.battery ]]; then
      if (( ac == 1 && (pct >= PAUSE_HIGH || watts >= BIG_ADAPTER) )); then
        rm -f PAUSE.battery PAUSE
        echo "[guard] $(date -Iseconds) battery ${pct}%, adapter ${watts:-0} W: resuming"
      fi
    else
      if (( pct <= PAUSE_LOW && (ac == 0 || watts < BIG_ADAPTER) )); then
        touch PAUSE PAUSE.battery
        echo "[guard] $(date -Iseconds) battery ${pct}%, adapter ${watts:-0} W: pausing until ${PAUSE_HIGH}% or a >=${BIG_ADAPTER} W adapter"
      fi
    fi
    sleep 60
  done
}
battery_guard &
GUARD=$!
trap 'kill $GUARD 2>/dev/null' EXIT

for i in {1..100}; do
  [[ -f STOP ]] && { echo "[loop] STOP file present, not restarting"; exit 0; }
  echo "[loop] start #$i $(date -Iseconds)"
  $PY /Users/mathiaskarlsson/Projects/smallestllm/data/gen_stories.py "$@"
  rc=$?
  if [[ $rc -eq 0 ]]; then echo "[loop] generator finished cleanly $(date -Iseconds)"; exit 0; fi
  echo "[loop] generator exited with $rc, restarting in 30 s"
  sleep 30
done
