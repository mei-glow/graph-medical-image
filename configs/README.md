These TOML files are model-specific run presets for the shared comparison pipeline.

Notes:
- They are "paper-aligned" only where the local repo/code clearly exposed model constraints such as input size and pretrained checkpoints.
- When the original paper training schedule was not available in the local files, the config keeps the project-wide comparison defaults instead of inventing paper hyperparameters.
- `train.py` and `eval.py` both accept `--config <path-to-toml>`.
- You still need to pass `--data-root` and, for evaluation, `--checkpoint` unless you add those fields to the TOML file yourself.

Example:
```bash
python -m semantic_attention.scripts.train --config semantic_attention/configs/transunet.toml --data-root semantic_attention/data --device cuda:0
python -m semantic_attention.scripts.eval --config semantic_attention/configs/transunet.toml --data-root semantic_attention/data --checkpoint outputs/transunet/best.pt --device cuda:0
```
