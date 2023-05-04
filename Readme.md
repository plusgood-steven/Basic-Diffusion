## Dataset

- Unzip MNIST.zip to `./data`
  ```sh
  unzip MNIST.zip -d ./data
  ```
- Folder structure
  ```
  .
  ├── data
  ├── dataset.py
  ├── Readme.md
  ├── environment.yml
  ├── interface.py
  ├── interface_ddim.py
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

### DDPM 
```sh
python interface.py --dir_path ./results --model_path /weight/best_model.pt
```

### DDIM 
```sh
python interface_ddim.py --dir_path ./results --model_path /weight/best_model.pt
```

The prediction images are in `results/output` folder.
The sample grid image is in `results/grid_output` folder.
## Eval

```sh
cd ./pytorch_gan_metrics.calc_metrics
python -m pytorch_gan_metrics.calc_metrics --path ../results/output --stats ../weight/mnist
```
The third number is FID
