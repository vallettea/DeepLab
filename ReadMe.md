# DeepWine

A small implementation of convNetJS using a [white wine quality data set](http://archive.ics.uci.edu/ml/datasets/Wine+Quality).

## Downloading the data:

```
make download
```

## Learn

```
make learn
```


## Some results

Globally for this dataset, **the prediction using the model projection on Principal Component space is more efficient than regular prediction** in terms of precision, provided the explained variance is acceptable (more than 80%).

However size reduction didn't affect computing time at all. This may be different for datasets with very large number of features, reduced into very few.

*RMSE: the mean of the distance between observations and predictions.*

*Pearson: ratio between (obs standard deviation * pred standard deviation) and obs pred covariance.*

### Without model projection:
* RMSE:  0.5989780261976226
* Pearson:  0.523834935300298
* Training time for 20 iterations on full dataset: 20.731 s
* Evaluating time 0.392 s

### With model projection on Principal Components space:
* Variance explained: 100%
* RMSE:  0.55296725885473
* Pearson:  0.6028430290047959
* Training time for 20 iterations 21.232 s
* Evaluation time 0.438 s

### With model projection on Principal Components subSpace:
**Model Size: 7**

* Variance explained: 87.9%
* RMSE:  0.5558405926044172
* Pearson:  0.5912188587309748
* Training time for 20 iterations 20.74 s
* Evaluation time 0.418 s

**Model Size: 3**
* Variance explained: 54.7%
* RMSE:  0.6224847521272912
* Pearson:  0.43953822719346686
* Training time for 20 iterations 19.613 s
* Evaluation time 0.413 s

**Model Size: 1**
* Variance explained: 29.3%
* RMSE:  0.658457924523246
* Pearson:  0.3101479092597284
* Training time for 20 iterations 20.666 s
* Evaluation time 0.432 s
