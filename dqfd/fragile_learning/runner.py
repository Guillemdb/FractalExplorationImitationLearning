from fragile.core import DiscreteUniform
from fragile.distributed import ReplayCreator
from plangym import AtariEnvironment

from dqfd.fragile_learning.env import AtariEnvironment


class FragileRunner:
    def __init__(
        self,
        game_name: str,
        n_swarms: int = 2,
        n_workers_per_swarm: int = 2,
        n_walkers: int = 32,
        max_epochs: int = 200,
        reward_scale: float = 2.0,
        distance_scale: float = 1.0,
        memory_size: int = 200,
        score_limit: float = 600,
    ):

        self.env = AtariEnvironment(game_name)
        self.n_actions = self.env.n_actions
        self.game_name = game_name
        self.env_callable = lambda: self.env
        self.model_callable = lambda env: DiscreteUniform(env=self.env)
        self.prune_tree = True
        self.memory_size = memory_size
        self.score_limit = score_limit
        # A bigger number will increase the quality of the trajectories sampled.
        self.n_walkers = n_walkers
        self.max_epochs = max_epochs  # Increase to sample longer games.
        self.reward_scale = reward_scale  # Rewards are more important than diversity.
        self.distance_scale = distance_scale
        self.minimize = False  # We want to get the maximum score possible.
        self.names = ["observs", "actions", "rewards", "oobs"]
        self.swarm = ReplayCreator(
            n_swarms=n_swarms,
            n_workers_per_swarm=n_workers_per_swarm,
            num_examples=self.memory_size,
            max_examples=int(self.memory_size * 1.5),
            model=self.model_callable,
            env=self.env_callable,
            names=self.names,
            n_walkers=self.n_walkers,
            max_epochs=self.max_epochs,
            prune_tree=self.prune_tree,
            reward_scale=self.reward_scale,
            distance_scale=self.distance_scale,
            minimize=self.minimize,
            score_limit=self.score_limit,
        )
        for name in self.names:
            setattr(self, name, None)

    def __len__(self):
        if getattr(self, self.names[0]) is None:
            return 0
        return len(getattr(self, self.names[0]))

    def iterate_memory(self):
        return self.swarm.iterate_values()

    def run(self):
        self.swarm.run()
        for name in self.names:
            setattr(self, name, getattr(self.swarm, name))


"""
swarm = ReplayCreator(
    names=names,
    num_examples=num_examples,
    max_examples=300,
    n_swarms=n_swarms, 
    n_workers_per_swarm=n_workers_per_swarm,
    model=model_callable,
    env=env_callable,
    n_walkers=n_walkers,
    max_epochs=max_epochs,
    reward_scale=reward_scale,
    distance_scale=distance_scale,
    minimize=minimize,
    force_logging=True,
    show_pbar=True, 
    report_interval=10,
)"""
