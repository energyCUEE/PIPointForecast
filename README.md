# PI Point Forecast

## Dependencies

This project is managed by [uv](https://docs.astral.sh/uv/). To install `uv`, run the appropriate command for your operating system:
```bash
# For Linux and macOS
curl -LsSf https://astral.sh/uv/install.sh | sh
```
```powershell
# For Windows PowerShell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

The project requires Python 3.13. You can install it using `uv`:
```bash
uv python install 3.13
```

After installing `uv` and Python 3.13, install the project dependencies by running:
```bash
uv sync
```

### Weights & Biases Environment (Optional)

For experiment tracking using [wandb](https://wandb.ai/), please copy `.env.example` to `.env` and fill in your `WANDB_API_KEY`. You can obtain an API key by creating an account on [wandb](https://wandb.ai/).

## Training

The training script is located at [`train.py`](train.py). It is managed by [Hydra](https://hydra.cc/), and the configuration files can be found in the [`config`](config) directory. 

For example, to train the `lstm_encdec` model for 1,000 epochs, run:
```bash
uv run train.py model=lstm_encdec run.max_epochs=1000 
```

Please refer to the [`config`](config) directory for more details on available configuration options.

## Evaluation

The evaluation scripts are located in the [`evaluation`](evaluation) directory. There are two notebooks for evaluation:

- [`evaluation.ipynb`](evaluation/evaluation.ipynb): Loads the pretrained model and evaluates it on the test set.
- [`eval_chronos_zeroshot.ipynb`](evaluation/eval_chronos_zeroshot.ipynb): Performs zero-shot evaluation for the Chronos-2 model on the test set without any fine-tuning.

## Repository Structure

```text
.
├── config/          # Hydra configuration files
├── data/            # Raw and processed datasets
├── evaluation/      # Jupyter notebooks for model evaluation
├── model/           # PyTorch neural network model definitions
├── save_model/      # Saved model checkpoints and weights
├── utils/           # Utility functions, metrics, and training helpers
├── gen_cached_data.py # Script to generate cached torch tensor from raw data
├── pyproject.toml
├── README.md
└── train.py         # Main training script
```

### Directory Details

- **`config/`**: Contains YAML configuration files managed by Hydra, including model hyperparameters, optimizers, schedulers, and trainer setups.
- **`data/`**: Stores dataframes and raw PyTorch tensor files (`.pt`) for training, validation, and testing.
- **`evaluation/`**: Includes Jupyter notebooks (e.g., `evaluation.ipynb`, `eval_chronos_zeroshot.ipynb`) and results for evaluating trained and zero-shot models.
- **`model/`**: Contains PyTorch implementations of the forecasting models.
- **`save_model/`**: Directory where the best model checkpoints (`.pth`) and their corresponding configurations are saved during training.
- **`utils/`**: Helper modules containing dataset loaders, loss functions, evaluation metrics, and the main trainer logic.
