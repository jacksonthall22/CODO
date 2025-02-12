import abc
from dataclasses import dataclass
from typing import Any, Type, TypeVar, Generic, cast
# Removed: import networkx as nx

# Generic type variables.
ParamsT = TypeVar("ParamsT")
ReturnT = TypeVar("ReturnT")


class Accumulator(Generic[ParamsT]):
    # An accumulator that stores each operation’s result.
    # It is keyed by the operation’s type, and __getitem__ casts the stored value
    # to the declared return type.
    _store: dict[Type["Operation[ParamsT, Any]"], Any]

    def __init__(self):
        self._store = {}

    def __getitem__(self, op_type: Type["Operation[ParamsT, ReturnT]"]) -> ReturnT:
        return cast(ReturnT, self._store[op_type])

    def __setitem__(self, op_type: Type["Operation[ParamsT, Any]"], value: Any) -> None:
        self._store[op_type] = value

    def __delitem__(self, op_type: Type["Operation[ParamsT, Any]"]) -> None:
        del self._store[op_type]

    def __repr__(self):
        return repr(self._store)

    def __str__(self):
        return str(self._store)


class Operation(abc.ABC, Generic[ParamsT, ReturnT]):
    # All operations share the same parameter container type.
    # Subclasses must override _call(), which receives an Accumulator and a parameters object.
    dependencies: list[Type["Operation[ParamsT, Any]"]] = []

    def __init__(self, silent: bool = False):
        self.silent = silent

    @abc.abstractmethod
    def __call__(self, acc: Accumulator[ParamsT], params: ParamsT) -> ReturnT: ...


class CODO(Generic[ParamsT]):
    # A container for operations. The type parameter ensures that every operation in the list
    # accepts the same parameter container.
    def __init__(self, operations: list[Operation[ParamsT, Any]]) -> None:
        self.operations = operations
        self.operations_by_type: dict[
            Type[Operation[ParamsT, Any]], Operation[ParamsT, Any]
        ] = {}
        for op in operations:
            op_type = type(op)
            if op_type in self.operations_by_type:
                raise ValueError(f"Duplicate operation type: {op_type}")
            self.operations_by_type[op_type] = op

    def _resolve_dependency_order(self) -> list[Type[Operation[ParamsT, Any]]]:
        # Create graph representation as adjacency list and in-degree count for each operation type.
        graph: dict[
            Type[Operation[ParamsT, Any]], list[Type[Operation[ParamsT, Any]]]
        ] = {}
        in_degree: dict[Type[Operation[ParamsT, Any]], int] = {}

        for op_type in self.operations_by_type:
            graph[op_type] = []
            in_degree[op_type] = 0

        for op_type, op in self.operations_by_type.items():
            for dep in getattr(op_type, "dependencies", []):
                if dep not in self.operations_by_type:
                    raise ValueError(f"Missing dependency: {dep} required by {op_type}")
                graph[dep].append(op_type)
                in_degree[op_type] += 1

        # Perform topological sort using Kahn's algorithm
        order: list[Type[Operation[ParamsT, Any]]] = []
        zero_in_degree = [node for node in in_degree if in_degree[node] == 0]

        while zero_in_degree:
            node = zero_in_degree.pop(0)
            order.append(node)
            for neighbor in graph[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    zero_in_degree.append(neighbor)

        if len(order) != len(self.operations_by_type):
            raise ValueError("Circular dependency detected")
        return order

    def __call__(
        self, params: ParamsT, *, with_silents: bool = False
    ) -> Accumulator[ParamsT]:
        order = self._resolve_dependency_order()
        acc: Accumulator[ParamsT] = Accumulator()
        for operation_type in order:
            operation = self.operations_by_type[operation_type]
            acc[operation_type] = operation(acc, params)

        if not with_silents:
            for operation_type in self.operations_by_type:
                if self.operations_by_type[operation_type].silent:
                    del acc[operation_type]

        return acc


if __name__ == "__main__":
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

    """
    {
        <class '__main__.Precision'>: 0.75,
        <class '__main__.Recall'>: 0.75,
        <class '__main__.Accuracy'>: 0.6666666666666666,
        <class '__main__.F1Score'>: 0.75
    }
    """
