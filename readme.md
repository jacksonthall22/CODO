# `CODO`: `C`ompute `O`perations in `D`ependency `O`rder

`CODO` is an easy-to-use library which abstracts the programming pattern where a set of operations (functions) with the same parameter lists must be computed over the same data, but where some of those operations might depend on other operations' outputs. 

A simple example is computing precision, recall, accuracy, and f1 scores for classification tasks. 

<img src="https://storage.googleapis.com/wandb-production.appspot.com/mostafaibrahim17/images/projects/37042936/05cc4b35.png?Expires=1739383979&GoogleAccessId=gorilla-files-url-signer-man%40wandb-production.iam.gserviceaccount.com&Signature=Yk0e6lnk2a96z%2FQbGY1vCOACWFqt8pIn%2BQ26v8OTF%2FvGqPjQ7sd6AaDittWRdbUjCOOCE7LngBJiVfl8%2FqAv0IY0VIdQtuwXD0jIne8OnMzMpYlV53ZLUgt9ABiLZ0EoSjzI54IfCvTOr1h57ssDPz%2FKkT136mSWOrLJZuIXLhu1D3d8MKt2AVabNtPXdO35zfV6eZesnKm1Tj2uQH7fChUTEms0PA8MkDdYJDnivHPuNSrQLZOk1wyuhaju%2B%2BW0V1G42atFFk272qR3V0Pk6KigqvVkEeXFKreEADhgaBF5f9WO7L00VnalrPmKKOXcTsSG2XatljMNiCJ5M1lLmw%3D%3D" />

All of these can be computed with their own functions that each take two arrays holding the true and predicted class labels, but then we'd be computing `TN`, `FN`, `FP`, and `FN` multiple times over (and we'd even have to re-compute precision and recall to get the f1 score). Alternatively, you could compute all these metrics in the scope of a single function, but for more complex tasks, it's often desirable to isolate this kind of logic into separate functions that each have one job.

`CODO` to the rescue: this library allows you to easily define these operations to isolate their logic with a simple API, automatically figure out the right order to run them, run the "shared" logic only once, and (optionally) parallelize where possible.


## How to use

1. 
    Define the params that each `Operation` will take. **Important:** All `Operation`s you plan to compute in the same `CODO` object must accept the same argument list. We wrap those params with a dataclass to enable full type safety. In our example, we'll define the types for the "true" and "predicted" class labels.

    ```python
    from dataclasses import dataclass

    @dataclass
    class MetricsParams:
        y_true: list[int]
        y_pred: list[int]
    ```

1. 
    Define operations by subclassing `codo.Operation`. The library defines this base class with two generic parameters that specify the input and output types for its `_call()` method, respectively (see code comments for details).

    Subclasses must override the `_call()` method, and may set a list of its `dependencies` as a class variable.

    The `_call()` method takes two parameters:

    - `acc`: A `codo.Accumulator[ParamsT]` object, where `ParamsT` is a generic variable that in our case represents our `MetricsParams` dataclass. Essentially this is a glorified `dict` providing full type safety that stores the result of operations computed earlier in the dependency tree. You can access those results by keying into `acc` with the operation's class.
    - `params`: An instance of the `MetricsParams` dataclass we defined above.

    ```python
    from codo import Operation
    from typing import override


    # When we subclass `Operation[MetricsParams, int]` below, the generics
    # are providing a shorthand for the following signature for the `_call()` method:
    #
    #     def _call(
    #         self,
    #         acc: codo.Accumulator[MetricsParams], 
    #         params: MetricsParams
    #     ) -> int:
    #         ...
    #
    # The first generic param sets the type for `params`,
    # and the second sets the method's return type.
    class TruePositives(Operation[MetricsParams, int]):
        @override
        def _call(self, acc, params):
            return sum(t == 1 and p == 1 for t, p in zip(true, pred))


    class FalsePositives(Operation[MetricsParams, int]):
        @override
        def _call(self, acc, params):
            return sum(t == 0 and p == 1 for t, p in zip(true, pred))


    class Precision(Operation[MetricsParams, float]):
        dependencies = [TruePositives, FalsePositives]

        @override
        def _call(self, acc, params):
            tp = acc[TruePositives]
            fp = acc[FalsePositives]
            # ^ Both resolve to `int` types here thanks to
            # generics on TruePositives/FalsePositives!

            return tp / (tp + fp)
    ```

1. 
    Create a `CODO` object with your `Operation`s, then call it like a function to get the outputs. 

    You may need to define `Operation`s for intermediate steps where you don't actually care about the output—in this case, you can set `silent=True` for that operation. This includes them in the dependency graph, but discards their output in the final returned dict unless `with_silents=True`.

    ```python
    from codo import CODO

    codo = CODO([
        Precision(),
        TruePositives(silent=True),
        FalsePositives(silent=True),
    ])
    # ^ Order doesn't matter in this list, Operations
    # will be computed in a valid dependency order.

    codo(
        MetricsParams(
            y_true=[1, 0, 1, 1, 0, 1],
            y_pred=[1, 1, 1, 0, 0, 1]
        ),
        # with_silents=True
    )

    # {<class '__main__.Precision'>: 0.75}
    ```

## Full example

The code below computes precision, recall, accuracy, and f1 scores.

```python
from typing import override

"""
Define parameter list for all Operations
"""

@dataclass
class MetricsParams:
    y_true: list[int]
    y_pred: list[int]

"""
Define your Operations
"""

class TruePositives(Operation[MetricsParams, int]):
    @override
    def __call__(self, acc, params):
        return sum(t == p == 1 for t, p in zip(params.y_true, params.y_pred))


class TrueNegatives(Operation[MetricsParams, int]):
    @override
    def __call__(self, acc, params):
        return sum(t == p == 0 for t, p in zip(params.y_true, params.y_pred))


class FalsePositives(Operation[MetricsParams, int]):
    @override
    def __call__(self, acc, params):
        return sum(t == 0 and p == 1 for t, p in zip(params.y_true, params.y_pred))


class FalseNegatives(Operation[MetricsParams, int]):
    @override
    def __call__(self, acc, params):
        return sum(t == 1 and p == 0 for t, p in zip(params.y_true, params.y_pred))


class Precision(Operation[MetricsParams, float]):
    dependencies = [TruePositives, FalsePositives]

    @override
    def __call__(self, acc, params):
        tp = acc[TruePositives]  # Resolves to `int` type
        fp = acc[FalsePositives]
        return tp / (tp + fp)


class Recall(Operation[MetricsParams, float]):
    dependencies = [TruePositives, FalseNegatives]

    @override
    def __call__(self, acc, params):
        tp = acc[TruePositives]
        fn = acc[FalseNegatives]
        return tp / (tp + fn)


class Accuracy(Operation[MetricsParams, float]):
    dependencies = [TruePositives, TrueNegatives, FalsePositives, FalseNegatives]

    @override
    def __call__(self, acc, params):
        tp = acc[TruePositives]
        tn = acc[TrueNegatives]
        fp = acc[FalsePositives]
        fn = acc[FalseNegatives]
        return (tp + tn) / (tp + tn + fp + fn)


class F1Score(Operation[MetricsParams, float]):
    dependencies = [Precision, Recall]

    @override
    def __call__(self, acc, params):
        precision = acc[Precision]
        recall = acc[Recall]
        return 2 * (precision * recall) / (precision + recall)


# Wrap all Operations in a CODO object. Defining the generic param type
# in-line allows you to query into `result` below with type safety.
# Dependency order doesn't matter in this list.
codo = CODO[MetricsParams](
    [
        Precision(),
        Recall(),
        Accuracy(),
        F1Score(),
        TruePositives(silent=True),
        TrueNegatives(silent=True),
        FalsePositives(silent=True),
        FalseNegatives(silent=True),
    ]
)

# Call your object like a function on arbitrary data
result = codo(
    MetricsParams(y_true=[1, 0, 1, 1, 0, 1], y_pred=[1, 1, 1, 0, 0, 1]),
    # with_silents=True,
)

# accuracy = result[Accuracy]  # Resolves to `float` type

print(result)

'''
{
    <class '__main__.Precision'>: 0.75,
    <class '__main__.Recall'>: 0.75,
    <class '__main__.Accuracy'>: 0.6666666666666666,
    <class '__main__.F1Score'>: 0.75
}
'''
```