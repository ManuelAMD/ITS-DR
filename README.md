# ITS-DR

ITS-DR is an interactive Python application for training and evaluating image time-series prediction models. It supports two approaches: autoencoder codification and spatial fragmentation.

## Requirements

- Windows, Python 3.12, and a working Tk installation (normally included with the standard Windows Python installer). The application uses Tk file dialogs.
- The packages listed in `requirements.txt` and Pillow, which is used to read and crop images.
- Enough memory and disk space for the image arrays and trained models. TensorFlow training can be resource-intensive. The autoencoder workflow asks whether to use a GPU, but its later training strategy may not honor that choice; device placement depends on the TensorFlow environment.

From PowerShell, at the repository root, install the dependencies:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt pillow
```

## Run

Run the application from the repository root. Relative paths in the code expect this working directory.

```powershell
python __main__.py
```

The application asks for a workflow in the terminal and opens Tk dialogs to select files and folders. It is not a command-line tool with command arguments.

### Train a model

Choose `1` to train the autoencoder approach or `2` to train the fragmentation approach. For either option, select how to load data:

- Choose `1` to load images. Select a CSV containing one image name per row, the source image directory, and a results directory. Images are read as grayscale PNGs. Enter the image dimensions, then answer the optional crop, mask, categorization, and save-as-NumPy prompts. The CSV order is the time-series order.
- Choose `2` to load an existing `.npy` array, then enter its rows, columns, and channels. The expected array layout is `(time_steps, rows, columns, channels)`; make its dimensions and channel count match the selected model configuration.

The default configurations are [Autoencoder_configuration.json](configurations/Autoencoder_configuration.json) and [Fragmentation_strategy.json](configurations/Fragmentation_strategy.json). They currently specify 120 rows, 360 columns, one grayscale channel, a seven-frame window, and a one-step prediction horizon. Edit the relevant JSON before running if your data differs. Configuration files are resolved from `configurations/` by filename.

Training saves model and prediction files in `Models/`, using names beginning with `DroughtDataset_model_testing_<timestamp>`. Autoencoder training also writes `Models/actual_decoder.h5`, which is needed to reconstruct predictions during evaluation. Preprocessing may write arrays such as `Models/x_test_data.npy` and `Models/y_test_data.npy`, plus `Mask.npy` when masking is enabled.

### Evaluate predictions

Choose `3`. Enter the shared prediction filename prefix (without `.npy`), select a configuration JSON from `configurations/`, enter the horizon and number of model parts, then load the evaluation dataset as a NumPy array when prompted. Select autoencoder or fragmentation evaluation. Fragmentation evaluation also asks for the number of fragments and largest filter size.

Evaluation expects numbered prediction files in `Models/`, such as `<prefix>_0.npy`, `<prefix>_1.npy`, and so on. The prefix must match the files produced by training. Autoencoder evaluation also requires `Models/actual_decoder.h5`. Evaluation output arrays are written to `Models/`.

Note: the current evaluation flow asks you to select a decoder file, but the selected path is not used; it loads `Models/actual_decoder.h5`. Keep the decoder at that exact path.

## Included data and artifacts

The repository includes sample CSV/NumPy data and previously generated model artifacts. Their presence does not mean every file is a ready-to-run input: use the NumPy-array dialog to choose the dataset and ensure it matches the dimensions in the configuration. New outputs are stored under `Models/` and may overwrite fixed-name evaluation or decoder artifacts.
