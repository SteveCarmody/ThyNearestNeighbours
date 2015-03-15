## Adaboost

We have also tried to perform the classification by using the Adaboost.M1 algorithm for multi-class classification. Specifically, we have used the implementation in the `R` package `adabag`. This ensemble approach uses classification trees as weak learners.

We have decided not to pursue this approach any further as the performance of this classifier was not satisfactory. On cross-validation tests, we were not able to achieve classification errors significantly below 30%. Moreover, even considering the large number of features, the execution time was disproportionately long.

