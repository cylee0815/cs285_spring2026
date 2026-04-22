# Running Offline RL Experiments

## 1. Local smoke test

Verify the full pipeline runs end-to-end in under a minute:

```bash
python offline_rl/scripts/run_offline.py --base_config=bc --smoke_test
python offline_rl/scripts/run_offline.py --base_config=iql --smoke_test
python offline_rl/scripts/run_offline.py --base_config=cql_vanilla --smoke_test
```

## 2. Single real run

```bash
python offline_rl/scripts/run_offline.py --base_config=iql --n_step=3 --seed=0 --run_group=offline_matrix
```

## 3. Generate full experiment matrix

```bash
python scripts/generate_offline_experiments.py
```

This creates `experiments/offline_experiments.txt` with 90 commands (10 algorithms x 3 n_step x 3 seeds).

### Batch execution (GNU parallel)

```bash
cat experiments/offline_experiments.txt | parallel -j 4
```

Or run a priority subset first (recommended):

```bash
grep -E "(iql|cql_vanilla|awac)" experiments/offline_experiments.txt | parallel -j 4
```

## 4. GPU cluster (SLURM)

Convert each command into a SLURM job:

```bash
while IFS= read -r cmd; do
  sbatch --gres=gpu:1 --mem=16G --time=4:00:00 --wrap="$cmd"
done < experiments/offline_experiments.txt
```

To split seeds across nodes:

```bash
# Node 1: seed=0 only
grep "seed=0" experiments/offline_experiments.txt | \
  while IFS= read -r cmd; do sbatch --wrap="$cmd"; done
```

To group by algorithm (useful for array jobs):

```bash
for algo in iql cql_vanilla awac; do
  grep "$algo" experiments/offline_experiments.txt > /tmp/${algo}_jobs.txt
  # Submit as a SLURM array
  N=$(wc -l < /tmp/${algo}_jobs.txt)
  sbatch --array=1-$N --wrap="sed -n \${SLURM_ARRAY_TASK_ID}p /tmp/${algo}_jobs.txt | bash"
done
```

## 5. Estimated compute

| Parameter      | Value                        |
|----------------|------------------------------|
| Total runs     | 90 (10 algos x 3 n_step x 3 seeds) |
| Est. per run   | ~15–60 min (GPU), ~1–4 hr (CPU) |
| Priority subset| IQL, CQL, AWAC (27 runs)     |

**Recommendation:** Run the priority subset first to validate the pipeline and get early signal before committing to the full matrix.

## Disabled algorithms

MBPO and MOPO are currently disabled (`NotImplementedError`) because their dynamics models assume raw features as observations, which is incompatible with the weight-augmented observation wrapper.
