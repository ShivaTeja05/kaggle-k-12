"""Budget estimate (E) for a Stage-B LoRA epoch per base model, scaled from the MEASURED Qwen3-8B candidate-student run
(runs/arch/student/pod_logs/logs.tgz: 8,505 rows, 1.686 rows/s steady state, RTX 6000 Ada $0.99/h, max 1,442 tokens).
Scaling assumption: time ~ active params; A100 ~1.7x RTX 6000 Ada bf16 dense throughput (assumption, not measured)."""
import json
rows, rps, ada = 8505, 1.686, 0.99
base_h = rows / rps / 3600
out = {'measured_qwen3_8b_epoch_hours_rtx6000ada': round(base_h, 2), 'measured_qwen3_8b_epoch_usd': round(base_h * ada, 2)}
for name, params, gpu, rate, speed in [('Qwen3.5-9B (bf16 LoRA ~22GB)', 9.7, 'RTX 6000 Ada 48GB', .99, 1.0),
                                       ('Gemma-4-31B / Qwen3.6-27B / Qwen3.8-27B (bf16 LoRA ~56GB)', 27.8, 'A100 80GB', 1.79, 1.7),
                                       ('Qwen3-32B QLoRA (current 32B recipe)', 32.8, 'A100 80GB', 1.79, 1.7)]:
    h = base_h * params / 8.2 / speed
    out[name] = {'gpu': gpu, 'epoch_hours_E': round(h, 2), 'epoch_usd_E': round(h * rate, 2),
                 'with_setup_0.5h_and_dev+test_generation_1h_E': round((h + 1.5) * rate, 2)}
out['note'] = ('Gated DeltaNet models (Qwen3.5/3.6/3.8) need flash-linear-attention + causal-conv1d kernels; without them the '
               'torch fallback can be several times slower (unmeasured). Gate any swap on a 100-step smoke throughput.')
json.dump(out, open('runs/deep/open-models/cost_estimate.json', 'w'), indent=1); print(json.dumps(out, indent=1))
