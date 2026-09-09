# Models

Three unsupervised detector families consume every frozen representation view. Preprocessing is fit on reference objects only, followed by model fitting on the same reference partition and scoring of held-out evaluation objects. In all interfaces, higher scores mean more unusual.

## Isolation Forest

Isolation Forest measures how readily an object is isolated by randomized trees. The publication grid varies tree count and feature subsampling; the score is `-score_samples(X)`. Reference objects alone are fitted.

```python
from stellar_anomaly_detection.models import IsolationForestDetector
```

## Local Outlier Factor

LOF compares local density with neighboring reference density. It runs in novelty mode so held-out objects can be scored without entering the fitted neighborhood structure. The publication score is `-score_samples(X_evaluation)`; reference `negative_outlier_factor_` values are diagnostic only.

```python
from stellar_anomaly_detection.models import LocalOutlierFactorDetector
```

## Autoencoder

The AE is a symmetric fully connected PyTorch network. It is trained only on reference objects and scores mean featurewise squared reconstruction error, with larger error interpreted as more unusual. Latent dimensions 2, 4, and 8 and three deterministic replications form the frozen sensitivity grid. PyTorch is optional, so import the implementation directly after installing the `ae` extra:

```python
from stellar_anomaly_detection.models.autoencoder import AutoencoderDetector
```

## Frozen configuration boundary

No best configuration is selected against evaluation objects. Fixed grids are aggregated under the frozen protocol, and each of the seven views is treated as an input representation rather than a claim of detector superiority. Exact configurations, architecture, scoring, seed policy, and fit boundaries are in the [final model configuration registry](../publication/supplementary/registries/final_model_configuration_registry.json).
