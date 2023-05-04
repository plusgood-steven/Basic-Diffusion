## Dataset

- Unzip data.zip to `./data`
  ```sh
  unzip data.zip -d ./data
  ```
- Folder structure
  ```
  .
  ├── data
  ├── dataset.py
  ├── Readme.md
  ├── environment.yml
  ├── interface.py
  ├── train.py
  ├── utils.py
  ├── model.py
  ├── pytorch-gan-metrics
  └── mnist.npz
  ```

## Environment

- using conda
  ```sh
  conda env create -f environment.yml
  ```

## Train

```sh
python train.py
```

## Interface

```sh
python interface.py --dir_path ./results --model_path /results/last_model.pt
```

The prediction images are in `results/output` folder.

## Eval

```sh
python -m pytorch_gan_metrics.calc_metrics --path /results/output --stats ./mnist
```
