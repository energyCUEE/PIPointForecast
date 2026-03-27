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