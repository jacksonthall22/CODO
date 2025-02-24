# `CODO`: (C)ompute (O)perations in (D)ependency (O)rder

`CODO` is an easy-to-use library which abstracts the programming pattern where a set of operations (functions) with the same parameter lists must be computed over the same data, but some of those operations might depend on others' outputs.

- **75** source lines of code
- **0** dependencies
- **Full type safety** thanks to `typing.Generic`

### What's the problem?
A common task* is computing `precision`, `recall`, `accuracy`, and `f1_score` for classification models in machine learning.

<sub>* Of course, tons of highly optimized libraries already solve this particular task, but it's a good way to demonstrate the problem `CODO` solves.</sub>

<img src="https://api.wandb.ai/files/mostafaibrahim17/images/projects/37042936/05cc4b35.png" />

Ideally, we want to compute these 4 metrics
- while isolating the logic for each computation
- without duplicating any work

To **isolate the logic**, we could compute each metric in its own function (each taking two arrays of true and predicted class labels), but then we'd be computing `TN`, `FN`, `FP`, and `FN` **multiple times over**, and we'd even have to re-compute `precision` and `recall` to get the `f1_score`.

To **avoid duplicating work**, we'd need to compute all metrics in a single block/function (or awkwardly save some "global" state in a parent scope), but then **we lose the pure logic isolation**. For more complex tasks, it's often desirable to keep functions simple where they each have exactly one job.

### `CODO` to the rescue!
- Define your operations with its simple API
- Specify their dependencies—other operations—to guarantee that their results will be accessable during its own computation (avoiding duplicate work)
- The library does the heavy lifting to automatically determine a valid computation graph and parallelize work wherever possible!*

<sub>* Parallelization not yet implemented, PRs welcome!</sub>


## Quickstart

### What we'll build

```python
codo = CODO[MetricsParams](
    [
        TruePositives(silent=True),
        TrueNegatives(silent=True),
        FalsePositives(silent=True),
        FalseNegatives(silent=True),
        Precision(),
        Recall(),
        Accuracy(),
        F1Score(),
    ]
)

y_true = [1, 0, 1, 1, 0, 1]
y_pred = [1, 1, 1, 0, 0, 1]

print(codo(MetricsParams(y_true, y_pred)))
'''
{
    <class '__main__.Precision'>: 0.75,
    <class '__main__.Recall'>: 0.75,
    <class '__main__.Accuracy'>: 0.6666666666666666,
    <class '__main__.F1Score'>: 0.75
}
'''
```

### Let's go!
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
    Define operations by subclassing `codo.Operation`. The library defines this base class with two generic parameters that specify the input and output types for its `__call__()` method, respectively (see code comments for details).

    Subclasses **must** override the `__call__()` method, and **may** set a list of its `dependencies` as a class variable.

    The `__call__()` method takes two parameters:

    - `acc`: A `codo.Accumulator[ParamsT]` object, where `ParamsT` is a generic variable that, in our case, will be the `MetricsParams` dataclass. Essentially this is a glorified `dict` providing full type safety that stores the results of operations computed earlier in the dependency tree. You can access those results by keying into `acc` with the operation's class.
    - `params`: An instance of the `MetricsParams` dataclass we defined above.

    <br />

    ```python
    from codo import Operation
    from typing import override


    # When we subclass `Operation[MetricsParams, int]` below, the generics
    # are providing a shorthand for the following signature for the `__call__()` method:
    #
    #     def __call__(
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
        def __call__(self, acc, params):
            return sum(t == 1 and p == 1 for t, p in zip(params.y_true, params.y_pred))


    class FalsePositives(Operation[MetricsParams, int]):
        @override
        def __call__(self, acc, params):
            return sum(t == 0 and p == 1 for t, p in zip(params.y_true, params.y_pred))


    class Precision(Operation[MetricsParams, float]):
        dependencies = [TruePositives, FalsePositives]

        @override
        def __call__(self, acc, params):
            tp = acc[TruePositives]
            fp = acc[FalsePositives]
            # ^ Both resolve to `int` types here thanks to
            # generics on TruePositives/FalsePositives!

            return tp / (tp + fp)
    ```

1. 
    Create a `CODO` object with your `Operation`s, then call it like a function to get the outputs.

    Notice that we specify the generic type in-line for the `CODO` object to get type safety on the `result` it returns, which is a `codo.Accumulator[MetricsParams]` object that has the same interface as a Python `dict`. This is where type safety gets cool, because `result[Precision]` will have type `float` and `result[TruePositives]` will have type `int`.

    You may have `Operation`s for intermediate steps where you don't actually care about the output—in those cases, you can set `silent=True` to discard their outputs, unless you set `with_silents=True`.

    ```python
    from codo import CODO

    codo = CODO[MetricsParams](
        [
            Precision(),
            TruePositives(silent=True),
            FalsePositives(silent=True),
        ]
    )
    # ^ Order doesn't matter in this list, Operations
    # will be computed in a valid dependency order.

    result = codo(
        MetricsParams(
            y_true=[1, 0, 1, 1, 0, 1],
            y_pred=[1, 1, 1, 0, 0, 1]
        ),
        # with_silents=True
    )

    print(result)
    # {<class '__main__.Precision'>: 0.75}
    ```

## Full example code

The code below computes precision, recall, accuracy, and f1 score. If you'd like, try to implement it yourself with the formulas and code above as a starting point, then come back here to check your answer!

```python
from typing import override

"""
Define parameter list for Operation types
"""

@dataclass
class MetricsParams:
    y_true: list[int]
    y_pred: list[int]

"""
Define all Operation types
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
# in-line allows you to query into `result` with type safety.
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

## Future work / `TODO`s
- [ ] Implement parallelize based on the dependency graph
- [ ] Visualize dependency graphs with `matplotlib` or `graphviz`
