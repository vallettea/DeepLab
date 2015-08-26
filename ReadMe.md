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

**Without model projection:**
Training time for 20 iterations on full dataset: 20.731 s

Final evaluation:
meanDistance:  0.5989780261976226
meanPearson:  0.523834935300298
Evaluating time 0.392 s

**With model projection on Principal Components space:**
Variance explained: 100%
Training time for 20 iterations 21.232 s

FINAL EVALUATION
meanDistance:  0.55296725885473
meanPearson:  0.6028430290047959
Evaluation time 0.438 s

**With model projection on Principal Components subSpace:**
*Model Size: 7*
Variance explained: 87.9%
Training time for 20 iterations 20.74 s

FINAL EVALUATION
meanDistance:  0.5558405926044172
meanPearson:  0.5912188587309748
Evaluation time 0.418 s

*Model Size: 3*
Variance explained: 54.7%
Training time for 20 iterations 19.613 s

FINAL EVALUATION
meanDistance:  0.6224847521272912
meanPearson:  0.43953822719346686
Evaluation time 0.413 s

*Model Size: 1*
Variance explained: 29.3%
Training time for 20 iterations 20.666 s

FINAL EVALUATION
meanDistance:  0.658457924523246
meanPearson:  0.3101479092597284
Evaluation time 0.432 s
