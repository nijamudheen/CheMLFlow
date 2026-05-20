# Quickstart

To get started with CheMLFlow, first install the package using the
[Installation](installation.md) guide. Once CheMLFlow is installed, you can train a
model on your own data or use the small bundled regression dataset to get a feel for the package.

Let's use the regression data that comes packaged with the training examples:

```bash
head MLModels/training/examples/regression.csv
```

```text
f1,f2,target
0.00,1.00,-0.50
0.10,1.10,-0.35
0.20,1.20,-0.20
0.30,1.30,-0.05
0.40,1.40,0.10
0.50,1.50,0.25
0.60,1.60,0.40
0.70,1.70,0.55
0.80,1.80,0.70
```

Now we're ready to train a simple random forest regression model:

```bash
python -m MLModels.training.cli train \
  --data-path MLModels/training/examples/regression.csv \
  --target-column target \
  --model-type random_forest \
  --task-type regression \
  --output-dir train_example
```

This trains a model on the regression dataset and saves the model artifacts, parameters, metrics,
and plots in the `train_example` directory. At the end, you should see output like:

```text
model: train_example/random_forest_best_model.pkl
params: train_example/random_forest_best_params.pkl
metrics: train_example/random_forest_metrics.json
```

With our trained model in hand, we can now use it to predict values for new rows. For demonstration
purposes, let's predict on the same rows that we used above:

```bash
python -m MLModels.training.cli predict \
  --test-path MLModels/training/examples/regression.csv \
  --target-column target \
  --model-path train_example/random_forest_best_model.pkl \
  --model-type random_forest \
  --task-type regression \
  --preds-path train_example/predictions.csv
```

This writes `train_example/predictions.csv`.

```bash
head train_example/predictions.csv
```

```text
f1,f2,target,pred_0
0.0,1.0,-0.5,-0.2615000000000002
0.1,1.1,-0.35,-0.2615000000000002
0.2,1.2,-0.2,-0.1894999999999998
0.3,1.3,-0.05,-0.049999999999999954
0.4,1.4,0.1,0.056499999999999974
0.5,1.5,0.25,0.22
0.6,1.6,0.4,0.40449999999999947
0.7,1.7,0.55,0.5274999999999992
0.8,1.8,0.7,0.6145
...
```

Given that this is a tiny demonstration dataset, the result is only meant to verify the workflow and
artifact layout. Use a larger train/test split or a DOE configuration for real model comparison.

In the rest of the documentation, we'll go into more detail about how to:

- Configure node-based pipelines
- Generate DOE batches
- Use local CSV, ChEMBL, and benchmark datasets
- Select feature inputs such as RDKit descriptors or Morgan fingerprints
- Train classical ML, deep learning, and foundation-model workflows
- Run analysis over completed experiments

## Summary

- Install CheMLFlow using the [Installation](installation.md) guide.
- Train a model with `python -m MLModels.training.cli train --data-path <input_path> --target-column <target> --model-type <model> --task-type <task> --output-dir <dir>`.
- Use a saved model for prediction with `python -m MLModels.training.cli predict --test-path <test_path> --target-column <target> --model-path <model_path> --model-type <model> --preds-path <path>`.
- Use DOE configs for full pipeline experiments across feature inputs, split strategies, and model families.
