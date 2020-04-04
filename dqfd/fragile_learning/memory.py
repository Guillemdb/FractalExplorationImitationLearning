import numpy

from dqfd.kerasrl.memory import PartitionedMemory as KrlPartitionedMemory

from dqfd.fragile_learning.runner import FragileRunner


class PartitionedMemory(KrlPartitionedMemory):
    def __init__(
        self,
        limit,
        runner: FragileRunner,
        alpha=0.4,
        start_beta=1.0,
        end_beta=1.0,
        steps_annealed=1,
        **kwargs
    ):
        def iterate_values(runner):
            if len(runner) == 0:
                raise ValueError("Memory is empty. Call memorize before iterating data.")
            for i in range(len(runner)):
                vals = [numpy.squeeze(getattr(runner, name)[i]) for name in runner.names]
                yield vals

        pre_load_data = list(iterate_values(runner))
        super(PartitionedMemory, self).__init__(
            pre_load_data=pre_load_data,
            limit=limit,
            alpha=alpha,
            start_beta=start_beta,
            end_beta=end_beta,
            steps_annealed=steps_annealed,
            **kwargs
        )
