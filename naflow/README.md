## Training

1. Download SIDD-Medium Dataset for training and generation:
```
python download_dataset.py
```

2. To train NAFlow with default settings, run
```
python main.py --config configs/train_naflow.yaml
```

## Reproduce the result

1. Download the pre-trained [model](https://1drv.ms/u/c/85cf5b7f538e2007/EQcgjlN_W88ggIU66gAAAAABfatiu1qFux_8HtaF4Ovliw?e=tNes3x) and place it in `./pretrained_models/`

2. Validate the pretrained NAFlow:
```
python main.py --config configs/validate_naflow.yaml
```

## Generation

1. Download the pre-trained [model](https://1drv.ms/u/c/85cf5b7f538e2007/EQcgjlN_W88ggIU66gAAAAABfatiu1qFux_8HtaF4Ovliw?e=tNes3x) and place it in `./pretrained_models/`

2. Generate the new noisy image with a sample image:
```
python scripts/generate_images.py
```