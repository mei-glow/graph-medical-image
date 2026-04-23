# Semantic Attention Research Pipeline


## Structure

```text
configs/
datasets/
engine/
models/
scripts/
utils/
```

## Dataset layout

Expected dataset root:

```text
data/
  ISIC2018_Task1-2_Training_Input/
  ISIC2018_Task1_Training_GroundTruth/
  ISIC2018_Task1-2_Validation_Input/
  ISIC2018_Task1_Validation_GroundTruth/
  ISIC2018_Task1-2_Test_Input/
  ISIC2018_Task1_Test_GroundTruth/
```

## Example usage

Train:

```bash
python -m scripts.train --model simple_unet --data-root semantic_attention/data --epochs 30
```

Evaluate:

```bash
python -m scripts.eval --model simple_unet --data-root semantic_attention/data --checkpoint outputs/simple_unet/best.pt
```

Benchmark multiple checkpoints:

```bash
python -m scripts.benchmark --data-root semantic_attention/data ^
  --run simple_unet=outputs/simple_unet/best.pt ^
  --run my_model=outputs/my_model/best.pt
```

## Plugging in your own models

Register any model in `models/builder.py`:

```python
@register_model("my_model")
def build_my_model(in_channels=3, out_channels=1, **kwargs):
    return MyModel(in_channels=in_channels, out_channels=out_channels, **kwargs)
```

All registered models automatically work with:

- `train.py`
- `eval.py`
- `benchmark.py`

as long as they return logits of shape `B x 1 x H x W` for binary segmentation.
