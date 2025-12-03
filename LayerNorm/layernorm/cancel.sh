for d in rs200_trial*/; do
  # pick the first slurm-*.out file in that directory
  f=$(printf "%s" "$d"/slurm-*.out 2>/dev/null | head -n1)

  # skip if no slurm file found
  [ -e "$f" ] || continue

  # extract jobid from slurm-5075958.out → 5075958
  jid=$(basename "$f" | sed -E 's/^slurm-([0-9]+)\.out$/\1/')

  scancel $jid
  # uncomment the next line when you're sure it's correct:
  # scancel "$jid"
done
