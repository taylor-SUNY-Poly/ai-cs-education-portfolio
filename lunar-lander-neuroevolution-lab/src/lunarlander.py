"""
lunarlander.py

Neuroevolution of a LunarLander controller using NEAT-Python and Gymnasium.

Usage
-----
Training (per phase)
    python lunarlander.py --phase1
    python lunarlander.py --phase2
    python lunarlander.py --phase3

Evaluate a saved winner genome
    python lunarlander.py --load <run_id> --episodes <n>

The loader expects genomes saved as
    genomes/winning_genome_<run_id>.pickle

For example
    python lunarlander.py --load 1.1 --episodes 5
will load genomes/winning_genome_1.1.pickle and render 5 episodes.

Each phase uses a different NEAT config file:
    phase 1 -> lunarlander_phase1.config
    phase 2 -> lunarlander_phase2.config
    phase 3 -> lunarlander_phase3.config

Artifacts are saved under phase directories:
    phase1/, phase2/, phase3/

Within each phase directory, per run:
    - results_phase{phase}.{run}.md
    - imgs/
        - fitness_{phase}.{run}.png
        - network_{phase}.{run}.png
        - rollout_{phase}.{run}.gif   (winner policy rollout, ~3 episodes)
    - genomes/
        - winning_genome_{phase}.{run}.pickle
"""

import argparse
import os
import sys
import time
import pickle
import multiprocessing
import re

import gymnasium as gym
import neat
import numpy as np

from report_utils import (
    get_next_run_index,
    plot_fitness_curves,
    plot_winner_network,
    write_markdown_summary,
    generate_winner_gif,
)

# Gymnasium environment ID (discrete Lunar Lander).
# For classic Gym this would be "LunarLander-v2"; for Gymnasium the canonical
# ID is now "LunarLander-v3".
ENV_ID = "LunarLander-v3"

# Global training environment (per-process, created lazily in eval_genome).
env = None


def make_env(render_mode: str | None = None):
    """
    Helper to create a LunarLander environment.

    Parameters
    ----------
    render_mode : str or None
        - None: headless (for training/eval)
        - "human": open a window for visualization
        - "rgb_array": off-screen rendering (for GIF/video)

    Returns
    -------
    gym.Env
        Newly created LunarLander-v3 environment.
    """
    return gym.make(ENV_ID, render_mode=render_mode)


def eval_genome(genome, config):
    """
    Evaluate a single genome by running it for several episodes and
    returning the mean total reward (fitness) across runs.

    This function is used by neat.ParallelEvaluator and will be called
    in worker processes.

    Parameters
    ----------
    genome : neat.Genome
        Genome being evaluated.
    config : neat.Config
        NEAT configuration object.

    Returns
    -------
    float
        Mean total reward over `runs_per_net` episodes.
    """
    global env

    # Lazily create an environment in this worker process if needed.
    if env is None:
        env = make_env(render_mode=None)

    # Build a feed-forward neural network from the genome.
    net = neat.nn.FeedForwardNetwork.create(genome, config)

    runs_per_net = 10
    episode_returns = []

    for _ in range(runs_per_net):
        # Gymnasium reset API: (observation, info)
        observation, _ = env.reset()
        total_reward = 0.0

        for _ in range(1000):
            # Forward pass through the network.
            action_values = net.activate(observation)
            action = int(np.argmax(action_values))

            # Gymnasium step API: (obs, reward, terminated, truncated, info)
            observation, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward

            if terminated or truncated:
                break

        # One fitness value per episode.
        episode_returns.append(total_reward)

    # The genome's fitness is the mean performance across all runs.
    return float(np.mean(episode_returns))


def eval_winner_net(winner, config, episodes: int = 100):
    """
    Evaluate the winner genome on a fresh environment for a number of
    episodes and report the mean fitness.

    Parameters
    ----------
    winner : neat.Genome
        Winning genome returned by NEAT.
    config : neat.Config
        NEAT configuration used during training.
    episodes : int, optional
        Number of episodes to evaluate, by default 100.

    Returns
    -------
    mean_fitness : float
        Average reward across evaluation episodes.
    eval_solved : bool
        True if mean_fitness >= 200.0 (LunarLander "solved" heuristic).
    """
    eval_env = make_env(render_mode=None)
    net = neat.nn.FeedForwardNetwork.create(winner, config)

    returns = []

    for _ in range(episodes):
        observation, _ = eval_env.reset()
        total_reward = 0.0

        for _ in range(1000):
            action_values = net.activate(observation)
            action = int(np.argmax(action_values))

            observation, reward, terminated, truncated, _ = eval_env.step(action)
            total_reward += reward

            if terminated or truncated:
                break

        returns.append(total_reward)

    eval_env.close()

    mean_fitness = float(np.mean(returns))
    print(f"Average fitness across {episodes} episodes: {mean_fitness:.2f}")

    # Gym/Gymnasium LunarLander is typically considered "solved" at mean >= 200.
    eval_solved = mean_fitness >= 200.0
    # Only surface the solved status for the canonical 100-episode, headless
    # evaluation used at the end of training runs. Short interactive viz runs
    # (e.g., via --load/--episodes) should stay quiet.
    if episodes == 100:
        if eval_solved:
            print(" + The task is solved + ")
        else:
            print(" - The task is not solved - ")

    return mean_fitness, eval_solved


def viz_winner_net(winner, config, episodes: int = 10):
    """
    Visualize the winner genome controlling the lander in a rendered window.

    Parameters
    ----------
    winner : neat.Genome
        Winning genome returned by NEAT.
    config : neat.Config
        NEAT configuration used during training.
    episodes : int, optional
        Number of visualization episodes, by default 10.

    Notes
    -----
    With Gymnasium's render API, passing render_mode="human" at creation
    time is sufficient. We generally should not call env.render() manually.
    """
    vis_env = make_env(render_mode="human")
    net = neat.nn.FeedForwardNetwork.create(winner, config)

    returns = []

    for episode in range(episodes):
        observation, _ = vis_env.reset()
        total_reward = 0.0

        for t in range(1000):
            action_values = net.activate(observation)
            action = int(np.argmax(action_values))

            observation, reward, terminated, truncated, _ = vis_env.step(action)
            total_reward += reward

            if terminated or truncated:
                print(
                    f"Episode {episode + 1}: "
                    f"return = {total_reward:.2f}, steps = {t + 1}"
                )
                break

        returns.append(total_reward)

    vis_env.close()

    if returns:
        mean_return = float(np.mean(returns))
        print(
            f"Average return across {len(returns)} rendered episodes: "
            f"{mean_return:.2f}"
        )

    return returns


def run(config_path: str, output_dir: str, phase: str, max_generations: int = 300):
    """
    Main NEAT training loop.

    Steps
    -----
    1. Load NEAT configuration.
    2. Create NEAT population and attach reporters.
    3. Run evolution with parallel evaluation until either:
       - the fitness_threshold in the config is reached, or
       - max_generations is hit.
    4. Evaluate the winner on fresh episodes.
    5. Save artifacts (fitness plot, network diagram, rollout GIF,
       winner genome, markdown summary) into `output_dir`.

    Parameters
    ----------
    config_path : str
        Path to the NEAT config file to use.
    output_dir : str
        Directory where all artifacts for this phase will be stored.
    phase : str
        Phase identifier ("1", "2", or "3"), used in filenames.
    max_generations : int, optional
        Maximum number of generations to run, by default 300.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Subdirectories for images and genomes
    imgs_dir = os.path.join(output_dir, "imgs")
    genomes_dir = os.path.join(output_dir, "genomes")
    os.makedirs(imgs_dir, exist_ok=True)
    os.makedirs(genomes_dir, exist_ok=True)

    # Load NEAT configuration.
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path,
    )

    # Create the population (top-level object for a NEAT run).
    population = neat.Population(config)

    # Attach reporters to show progress in the terminal and record stats.
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)
    population.add_reporter(neat.StdOutReporter(True))

    # Parallel execution over all available CPUs.
    pe = neat.ParallelEvaluator(multiprocessing.cpu_count(), eval_genome)

    # Measure training time (population.run only).
    start_time = time.time()
    winner = population.run(pe.evaluate, n=max_generations)
    train_time_sec = time.time() - start_time

    print("\nBest genome:\n", winner)

    # Determine run index for this phase based on existing images.
    run_index = get_next_run_index(imgs_dir, phase)

    # Evaluate the evolved network on multiple episodes.
    eval_mean, eval_solved = eval_winner_net(
        winner, config, episodes=100
    )

    # Save the winning genome in genomes subfolder with the requested naming.
    genome_filename = f"winning_genome_{phase}.{run_index}.pickle"
    genome_path = os.path.join(genomes_dir, genome_filename)
    with open(genome_path, "wb") as f:
        pickle.dump(winner, f)
    print(f"Saved winning genome to {genome_path}")
    # Relative path used in markdown for convenience.
    genome_rel_path = f"genomes/{genome_filename}"

    # Plot fitness curves in imgs subfolder.
    plot_fitness_curves(
        statistics=stats,
        imgs_dir=imgs_dir,
        phase=phase,
        run_index=run_index,
    )

    # Plot winner network architecture in imgs subfolder.
    plot_winner_network(
        winner=winner,
        config=config,
        imgs_dir=imgs_dir,
        phase=phase,
        run_index=run_index,
    )

    # Generate rollout GIF via report_utils (uses default settings).
    gif_rel_path = generate_winner_gif(
        winner=winner,
        config=config,
        imgs_dir=imgs_dir,
        phase=phase,
        run_index=run_index,
        env_factory=make_env,
    )

    # Write markdown summary (includes links to plots, GIF, and genome config).
    write_markdown_summary(
        statistics=stats,
        output_dir=output_dir,
        phase=phase,
        run_index=run_index,
        winner=winner,
        eval_mean_fitness=eval_mean,
        eval_solved=eval_solved,
        train_time_sec=train_time_sec,
        fitness_threshold=config.fitness_threshold,
        eval_episodes=100,
        config=config,
        genome_rel_path=genome_rel_path,
        gif_rel_path=gif_rel_path,
    )

    # Visualize a few episodes (human render).
    viz_winner_net(winner, config, episodes=10)

    # Clean up the shared training env in this process, if it was created.
    global env
    if env is not None:
        env.close()
        env = None


def _build_argument_parser() -> argparse.ArgumentParser:
    """Create an argument parser for the CLI interface."""

    parser = argparse.ArgumentParser(
        description="Train or visualize NEAT agents for LunarLander-v3."
    )
    parser.add_argument(
        "--load",
        metavar="RUN_ID",
        help=(
            "Identifier of a saved genome to visualize. Accepts either the "
            "<phase>.<run> shorthand (e.g. 1.1) or a direct path to a pickle file."
        ),
    )
    parser.add_argument(
        "--episodes",
        metavar="N",
        type=int,
        help="Number of episodes to evaluate or visualize a saved genome.",
    )

    phase_group = parser.add_mutually_exclusive_group()
    phase_group.add_argument(
        "--phase",
        choices=["1", "2", "3"],
        help="Train the experiment for the specified phase (1, 2, or 3).",
    )
    phase_group.add_argument(
        "--phase1",
        action="store_true",
        help="Shortcut for training phase 1.",
    )
    phase_group.add_argument(
        "--phase2",
        action="store_true",
        help="Shortcut for training phase 2.",
    )
    phase_group.add_argument(
        "--phase3",
        action="store_true",
        help="Shortcut for training phase 3.",
    )

    return parser


def _parse_phase_from_argv(argv):
    """
    Parse the phase from command-line arguments.

    Expected usage
    --------------
        python lunarlander.py 1
        python lunarlander.py 2
        python lunarlander.py 3

    Also accepts
        python lunarlander.py phase1
        python lunarlander.py phase2
        python lunarlander.py phase3

    Parameters
    ----------
    argv : list[str]
        Command-line arguments (typically sys.argv).

    Returns
    -------
    str
        Phase string: "1", "2", or "3".

    Raises
    ------
    SystemExit
        If the phase argument is missing or invalid.
    """
    if len(argv) < 2:
        raise SystemExit(
            "Usage: python lunarlander.py <phase>\n"
            "  where <phase> is 1, 2, or 3"
        )

    phase = argv[1].strip().lower()

    # Allow "phase1" style input by stripping "phase" prefix.
    if phase.startswith("phase"):
        phase = phase.replace("phase", "")

    if phase not in {"1", "2", "3"}:
        raise SystemExit(
            f"Invalid phase '{argv[1]}'. Expected 1, 2, or 3."
        )

    return phase


def _infer_phase_from_genome_filename(filename: str) -> str:
    """Infer the training phase from a saved genome filename.

    Parameters
    ----------
    filename : str
        Basename or path to a genome file saved via ``run``.

    Returns
    -------
    str
        Phase identifier ("1", "2", or "3").

    Raises
    ------
    SystemExit
        If the phase cannot be inferred from the filename.
    """

    basename = os.path.basename(filename)
    match = re.search(r"winning_genome_(\d)\.", basename)
    if match:
        return match.group(1)

    raise SystemExit(
        "Unable to infer phase from genome filename. "
        "Expected pattern 'winning_genome_<phase>.<run>.pickle'."
    )


def _resolve_genome_path(base_dir: str, genome_identifier: str) -> tuple[str, str]:
    """Resolve a genome identifier to an on-disk path and phase string.

    Parameters
    ----------
    base_dir : str
        Directory containing the script and config files.
    genome_identifier : str
        Either a direct path to a pickle file or the ``<run_id>`` suffix used
        during training (e.g. ``1.1``).

    Returns
    -------
    tuple[str, str]
        A tuple of ``(path, phase)`` where ``path`` is the absolute path to the
        genome pickle and ``phase`` is the inferred phase string.
    """

    if genome_identifier.endswith(".pickle") or os.sep in genome_identifier:
        candidate = genome_identifier
        if not os.path.isabs(candidate):
            candidate = os.path.join(base_dir, candidate)
        if not os.path.exists(candidate):
            raise SystemExit(f"Genome file not found: {candidate}")
        phase = _infer_phase_from_genome_filename(candidate)
        return candidate, phase

    genome_filename = f"winning_genome_{genome_identifier}.pickle"
    # Primary location: genomes/ directory adjacent to this script.
    genomes_dir = os.path.join(base_dir, "genomes")
    candidate_paths = [os.path.join(genomes_dir, genome_filename)]

    # Fall back to historical location phase*/genomes/.
    try:
        phase_from_id = genome_identifier.split(".")[0]
    except IndexError:
        phase_from_id = ""

    if phase_from_id in {"1", "2", "3"}:
        candidate_paths.append(
            os.path.join(base_dir, f"phase{phase_from_id}", "genomes", genome_filename)
        )

    for candidate in candidate_paths:
        if os.path.exists(candidate):
            phase = _infer_phase_from_genome_filename(candidate)
            return candidate, phase

    raise SystemExit(
        "Could not locate genome file. Checked: " + ", ".join(candidate_paths)
    )


def _load_config_for_phase(base_dir: str, phase: str) -> neat.Config:
    """Load the NEAT config corresponding to a given phase."""

    config_filename = f"lunarlander_phase{phase}.config"
    config_path = os.path.join(base_dir, config_filename)
    if not os.path.exists(config_path):
        raise SystemExit(f"Config file not found for phase {phase}: {config_path}")

    return neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path,
    )


def run_saved_policy(
    genome_identifier: str, episodes: int, *, render: bool = False
) -> None:
    """Load a saved genome and evaluate or visualize it."""

    if episodes <= 0:
        raise SystemExit("Episodes must be a positive integer.")

    base_dir = os.path.dirname(__file__)
    genome_path, phase = _resolve_genome_path(base_dir, genome_identifier)
    print(f"Loading genome from: {genome_path}")

    with open(genome_path, "rb") as f:
        winner = pickle.load(f)

    config = _load_config_for_phase(base_dir, phase)

    if render:
        print(
            f"Rendering saved policy for phase {phase} across {episodes} episodes..."
        )
        viz_winner_net(winner, config, episodes=episodes)
    else:
        print(
            f"Evaluating saved policy for phase {phase} across {episodes} episodes..."
        )
        eval_winner_net(winner, config, episodes=episodes)


def main():
    """
    Entry point for running a NEAT LunarLander experiment for a given phase.

    Reads the phase argument, resolves the matching config file, and
    dispatches to `run`.
    """
    local_dir = os.path.dirname(__file__)
    argv = sys.argv

    # Legacy positional CLI support (pre-flag interface).
    if len(argv) >= 2 and not argv[1].startswith("-"):
        if argv[1].lower() == "load":
            if len(argv) < 5 or argv[3].lower() != "episodes":
                raise SystemExit(
                    "Usage: python lunarlander.py load <run_id> episodes <n>"
                )

            genome_identifier = argv[2]
            try:
                episodes = int(argv[4])
            except ValueError as exc:
                raise SystemExit("Episodes must be an integer.") from exc

            run_saved_policy(genome_identifier, episodes, render=True)
            return

        phase = _parse_phase_from_argv(argv)
    else:
        parser = _build_argument_parser()
        args = parser.parse_args(argv[1:])

        if args.load:
            if args.episodes is None:
                parser.error("--episodes is required when using --load.")
            if args.episodes <= 0:
                parser.error("--episodes must be a positive integer.")
            if any([args.phase, args.phase1, args.phase2, args.phase3]):
                parser.error("--load cannot be combined with phase training options.")

            run_saved_policy(args.load, args.episodes, render=True)
            return

        if args.episodes is not None:
            parser.error("--episodes is only valid when using --load.")

        phase = None
        if args.phase:
            phase = args.phase
        elif args.phase1:
            phase = "1"
        elif args.phase2:
            phase = "2"
        elif args.phase3:
            phase = "3"

        if phase is None:
            parser.error(
                "Specify one of --phase, --phase1, --phase2, or --phase3, "
                "or use --load to visualize a saved policy."
            )

    # Map phase number to config file name.
    config_filename = {
        "1": "lunarlander_phase1.config",
        "2": "lunarlander_phase2.config",
        "3": "lunarlander_phase3.config",
    }[phase]

    config_file = os.path.join(local_dir, config_filename)

    # Output directory per phase: phase1/, phase2/, phase3/.
    output_dir = os.path.join(local_dir, f"phase{phase}")

    print(f"Running phase {phase} with config: {config_file}")
    print(
        f"Artifacts (markdown, images, GIFs, genomes) "
        f"will be saved in: {output_dir}"
    )

    run(config_file, output_dir, phase)


if __name__ == "__main__":
    main()
