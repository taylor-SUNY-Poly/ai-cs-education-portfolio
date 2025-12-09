"""
report_utils.py

Utility functions for reporting and visualization for NEAT-based
LunarLander experiments.

Responsibilities
----------------
- Determine per-phase run indices based on existing plots.
- Plot fitness curves over generations.
- Plot winner network architecture (nodes and connections).
- Generate rollout GIFs of the winner policy.
- Write a markdown report summarizing each run, including:
    * Training and evaluation statistics
    * Links to plots and rollout GIF
    * Structured tables for nodes and connections in the winner genome

These utilities are imported and used by `lunarlander.py`.
"""

import os
import re
from typing import Callable, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import neat
import imageio.v2 as imageio

def get_next_run_index(imgs_dir: str, phase: str) -> int:
    """
    Determine the next run index for a given phase by scanning existing
    fitness_{phase}.{run}.png files in the `imgs_dir` and incrementing
    the highest run number.

    Parameters
    ----------
    imgs_dir : str
        Path to the images subdirectory for this phase.
    phase : str
        Phase identifier ("1", "2", or "3"), used in the filename pattern.

    Returns
    -------
    int
        Next run index to use (1-based).
    """
    # Regex to match filenames like "fitness_1.3.png"
    pattern = re.compile(rf"fitness_{phase}\.(\d+)\.png$")
    max_run = 0

    if os.path.isdir(imgs_dir):
        for fname in os.listdir(imgs_dir):
            match = pattern.match(fname)
            if match:
                try:
                    run_num = int(match.group(1))
                except ValueError:
                    continue
                max_run = max(max_run, run_num)

    # Next run is highest existing + 1 (or 1 if no plots exist yet).
    return max_run + 1


def plot_fitness_curves(
    statistics: neat.StatisticsReporter,
    imgs_dir: str,
    phase: str,
    run_index: int,
    filename: str | None = None,
) -> None:
    """
    Create and save a fitness plot (best, average, ±1 stdev over generations).

    The output file is named `fitness_{phase}.{run_index}.png` by default and
    is written into `imgs_dir`.

    Parameters
    ----------
    statistics : neat.StatisticsReporter
        Statistics reporter attached to the NEAT population.
    imgs_dir : str
        Directory where image files are stored for this phase.
    phase : str
        Phase identifier ("1", "2", or "3").
    run_index : int
        Run index for this phase (1-based).
    filename : str or None, optional
        Custom filename (no path). If None, a default is used.
    """
    if filename is None:
        filename = f"fitness_{phase}.{run_index}.png"

    generations = range(len(statistics.most_fit_genomes))
    generations_list = list(generations)

    best_fitness = [g.fitness for g in statistics.most_fit_genomes]
    avg_fitness = statistics.get_fitness_mean()
    stdev_fitness = statistics.get_fitness_stdev()

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot best and average fitness.
    ax.plot(generations_list, best_fitness, label="Best fitness")
    ax.plot(generations_list, avg_fitness, label="Average fitness")

    # Plot shaded region for ±1 standard deviation around the mean.
    avg_arr = np.array(avg_fitness, dtype=float)
    stdev_arr = np.array(stdev_fitness, dtype=float)
    ax.fill_between(
        generations_list,
        avg_arr - stdev_arr,
        avg_arr + stdev_arr,
        alpha=0.2,
        label="±1 stdev",
    )

    ax.set_title(f"Phase {phase} - Run {run_index} - Fitness over Generations")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.3)

    fig.tight_layout()
    filepath = os.path.join(imgs_dir, filename)
    fig.savefig(filepath)
    plt.close(fig)
    print(f"Saved fitness plot to {filepath}")


def plot_winner_network(
    winner,
    config: neat.Config,
    imgs_dir: str,
    phase: str,
    run_index: int,
    filename: str | None = None,
) -> None:
    """
    Plot a simple network diagram of the winner genome and save it to disk.

    Layout:
        - Input nodes at x = 0
        - Hidden nodes at x = 1
        - Output nodes at x = 2
    Connections are drawn with color and thickness based on their weight.

    Parameters
    ----------
    winner : neat.DefaultGenome
        Winning genome returned by NEAT.
    config : neat.Config
        NEAT configuration used for this run.
    imgs_dir : str
        Directory where image files are stored for this phase.
    phase : str
        Phase identifier ("1", "2", or "3").
    run_index : int
        Run index for this phase (1-based).
    filename : str or None, optional
        Custom filename (no path). If None, a default is used.
    """
    if filename is None:
        filename = f"network_{phase}.{run_index}.png"

    genome_config = config.genome_config

    # Input and output node IDs are given by the genome configuration.
    input_keys = list(genome_config.input_keys)
    output_keys = list(genome_config.output_keys)

    # Hidden nodes are any nodes in the genome which are not inputs or outputs.
    hidden_keys = [
        k for k in winner.nodes.keys()
        if k not in input_keys and k not in output_keys
    ]

    input_keys_sorted = sorted(input_keys)
    output_keys_sorted = sorted(output_keys)
    hidden_keys_sorted = sorted(hidden_keys)

    # Assign 2D positions for each node: (x, y)
    node_positions: Dict[int, Tuple[float, float]] = {}

    def assign_layer_positions(keys: List[int], x_pos: float) -> None:
        """
        Assign vertical positions for a list of node keys at a given x coordinate.

        Nodes are evenly spaced between y=0 and y=1.
        """
        if not keys:
            return
        if len(keys) == 1:
            # Single node: place at vertical center.
            node_positions[keys[0]] = (x_pos, 0.5)
            return
        for idx, key in enumerate(keys):
            y = idx / (len(keys) - 1)
            node_positions[key] = (x_pos, y)

    # Place each layer of nodes.
    assign_layer_positions(input_keys_sorted, x_pos=0.0)
    assign_layer_positions(hidden_keys_sorted, x_pos=1.0)
    assign_layer_positions(output_keys_sorted, x_pos=2.0)

    fig, ax = plt.subplots(figsize=(8, 6))

    # Draw connections first (under the nodes).
    for (i, j), conn_gene in winner.connections.items():
        if not conn_gene.enabled:
            continue
        if i not in node_positions or j not in node_positions:
            continue

        x1, y1 = node_positions[i]
        x2, y2 = node_positions[j]

        # Line width and color encode weight sign and magnitude.
        weight = conn_gene.weight
        line_width = 0.5 + 1.5 * min(1.0, abs(weight) / 5.0)
        color = "green" if weight > 0 else "red"

        ax.plot(
            [x1, x2],
            [y1, y2],
            linewidth=line_width,
            color=color,
            alpha=0.7,
            zorder=1,
        )

    # Helper to draw node markers.
    def draw_nodes(keys: List[int], facecolor: str, label: str) -> None:
        if not keys:
            return
        xs = [node_positions[k][0] for k in keys]
        ys = [node_positions[k][1] for k in keys]
        ax.scatter(
            xs,
            ys,
            s=300,
            c=facecolor,
            edgecolors="black",
            zorder=2,
            label=label,
        )
        for k, x, y in zip(keys, xs, ys):
            ax.text(
                x,
                y,
                str(k),
                ha="center",
                va="center",
                fontsize=8,
                color="black",
                zorder=3,
            )

    # Draw input, hidden, and output nodes with different colors.
    draw_nodes(input_keys_sorted, facecolor="lightblue", label="Inputs")
    draw_nodes(hidden_keys_sorted, facecolor="lightgray", label="Hidden")
    draw_nodes(output_keys_sorted, facecolor="lightgreen", label="Outputs")

    ax.set_title(f"Phase {phase} - Run {run_index} - Winner Network Architecture")
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(-0.1, 1.1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc="upper right", fontsize="small")
    ax.set_axis_off()

    fig.tight_layout()
    filepath = os.path.join(imgs_dir, filename)
    fig.savefig(filepath)
    plt.close(fig)
    print(f"Saved network diagram to {filepath}")


def generate_winner_gif(
    winner,
    config: neat.Config,
    imgs_dir: str,
    phase: str,
    run_index: int,
    env_factory: Callable[[str | None], object],
    episodes: int = 3,
    max_steps: int = 600,
    fps: int = 30,
    frame_subsample: int = 5,
) -> str | None:
    """
    Roll out the winner policy for a small number of episodes and save
    the frames as an animated GIF.

    To keep the GIF from feeling too slow, we subsample the captured
    frames (by default keep every 5th frame), which makes the motion
    appear faster while keeping the same per-frame display duration.

    Parameters
    ----------
    winner : neat.Genome
        Winning genome to visualize.
    config : neat.Config
        NEAT configuration used for this run.
    imgs_dir : str
        Directory where image files are stored for this phase.
    phase : str
        Phase identifier ("1", "2", or "3").
    run_index : int
        Run index for this phase (1-based).
    env_factory : callable
        A function taking `render_mode: str | None` and returning a Gymnasium
        environment. This decouples reporting from the specific environment
        implementation.
    episodes : int, optional
        Number of episodes to record (default 3).
    max_steps : int, optional
        Maximum steps per episode (default 600).
    fps : int, optional
        Target frames-per-second for playback (default 30).
    frame_subsample : int, optional
        Keep one out of every `frame_subsample` frames (>=1). For example:
        - 1  -> keep all frames
        - 2  -> keep every other frame
        - 5  -> keep every 5th frame (default)

    Returns
    -------
    str or None
        Relative path to the GIF (e.g. "imgs/rollout_1.3.gif") if created
        successfully, or None if GIF generation was skipped or failed.
    """
    if imageio is None:
        print("imageio is not installed; skipping GIF generation.")
        return None

    if frame_subsample < 1:
        frame_subsample = 1

    gif_filename = f"rollout_{phase}.{run_index}.gif"
    gif_path = os.path.join(imgs_dir, gif_filename)

    # Environment that returns RGB arrays for rendering.
    env = env_factory("rgb_array")
    net = neat.nn.FeedForwardNetwork.create(winner, config)

    frames: List[np.ndarray] = []

    for ep in range(episodes):
        observation, _ = env.reset()
        # Capture an initial frame if available.
        frame = env.render()
        if frame is not None:
            frames.append(frame)

        for t in range(max_steps):
            action_values = net.activate(observation)
            action = int(np.argmax(action_values))

            observation, reward, terminated, truncated, _ = env.step(action)

            frame = env.render()
            if frame is not None:
                frames.append(frame)

            if terminated or truncated:
                break

    env.close()

    if not frames:
        print("No frames captured; GIF will not be created.")
        return None

    # Subsample frames to speed up apparent motion.
    frames_subsampled = frames[::frame_subsample]
    if len(frames_subsampled) < 2:
        # Fallback: need at least a couple of frames, otherwise use original.
        frames_subsampled = frames

    # Convert fps to per-frame duration (in seconds) and loop infinitely.
    frame_duration = 1.0 / fps if fps > 0 else 1.0 / 30.0
    imageio.mimsave(gif_path, frames_subsampled, duration=frame_duration, loop=0)
    print(f"Saved rollout GIF to {gif_path}")
    # Markdown will reference this relative to the phase directory.
    return f"imgs/{gif_filename}"


def write_markdown_summary(
    statistics: neat.StatisticsReporter,
    output_dir: str,
    phase: str,
    run_index: int,
    winner,
    eval_mean_fitness: float,
    eval_solved: bool,
    train_time_sec: float,
    fitness_threshold: float,
    eval_episodes: int,
    config: neat.Config,
    genome_rel_path: str,
    gif_rel_path: str | None,
) -> None:
    """
    Write a markdown file summarizing the run, including:

    - Training & evaluation statistics
    - Links to fitness and network PNGs
    - Optional rollout GIF if available
    - Winner genome configuration:
        * Node table: type, activation, bias
        * Connection table: from, to, weight, enabled

    The markdown file is named:
        results_phase{phase}.{run_index}.md

    Parameters
    ----------
    statistics : neat.StatisticsReporter
        Statistics reporter attached to the NEAT population.
    output_dir : str
        Phase output directory in which the markdown will be saved.
    phase : str
        Phase identifier ("1", "2", or "3").
    run_index : int
        Run index for this phase (1-based).
    winner : neat.DefaultGenome
        Winning genome returned by NEAT.
    eval_mean_fitness : float
        Mean fitness across evaluation episodes.
    eval_solved : bool
        True if eval_mean_fitness >= 200.0.
    train_time_sec : float
        Total time spent in population.run(), in seconds.
    fitness_threshold : float
        Fitness threshold from the NEAT config (for early stopping).
    eval_episodes : int
        Number of evaluation episodes used.
    config : neat.Config
        NEAT configuration used for this run.
    genome_rel_path : str
        Relative path (from `output_dir`) to the pickled winning genome file.
    gif_rel_path : str or None
        Relative path to the rollout GIF (from `output_dir`), or None if
        GIF generation was skipped.
    """
    md_filename = f"results_phase{phase}.{run_index}.md"
    md_path = os.path.join(output_dir, md_filename)

    num_generations = len(statistics.most_fit_genomes)
    train_solved = (winner.fitness is not None) and (
        winner.fitness >= fitness_threshold
    )

    # Genome-level structure info.
    genome_config = config.genome_config
    input_keys = list(genome_config.input_keys)
    output_keys = list(genome_config.output_keys)
    # In NEAT-Python, winner.nodes typically holds hidden and output nodes.
    hidden_keys = [
        k for k in winner.nodes.keys()
        if k not in input_keys and k not in output_keys
    ]
    input_keys_sorted = sorted(input_keys)
    output_keys_sorted = sorted(output_keys)
    hidden_keys_sorted = sorted(hidden_keys)

    num_inputs = len(input_keys_sorted)
    num_outputs = len(output_keys_sorted)
    num_hidden = len(hidden_keys_sorted)

    enabled_connections = [
        cg for cg in winner.connections.values() if cg.enabled
    ]
    num_enabled_conns = len(enabled_connections)

    # Relative paths to images and genome for markdown.
    png_fitness_rel = f"imgs/fitness_{phase}.{run_index}.png"
    png_network_rel = f"imgs/network_{phase}.{run_index}.png"

    lines: List[str] = []

    # Top-level heading.
    lines.append(f"## LunarLander NEAT Results – Phase {phase}, Run {run_index}\n")

    # Training statistics.
    lines.append("### Training Summary")
    lines.append(f"- Generations run: **{num_generations}**")
    lines.append(f"- Training fitness threshold: **{fitness_threshold:.1f}**")
    lines.append(
        f"- Winner training fitness (runs_per_net=10): **{winner.fitness:.2f}**"
    )
    lines.append(
        f"- Training solved (by threshold)? **{'YES' if train_solved else 'NO'}**"
    )
    lines.append(
        f"- Training time (population.run): **{train_time_sec:.1f} seconds**\n"
    )

    # Evaluation statistics.
    lines.append("### Evaluation Summary")
    lines.append(f"- Evaluation episodes: **{eval_episodes}**")
    lines.append(f"- Evaluation mean fitness: **{eval_mean_fitness:.2f}**")
    lines.append(
        f"- Evaluation solved (mean ≥ 200)? **{'YES' if eval_solved else 'NO'}**\n"
    )

    # Fitness plot.
    lines.append("### Fitness Plot")
    lines.append(f"![Fitness over generations]({png_fitness_rel})\n")

    # Winning Genome Visualization (GIF) right after fitness plot.
    if gif_rel_path is not None:
        lines.append("### Winning Genome Visualization")
        lines.append(f"![Winner rollout]({gif_rel_path})\n")

    # Winner network diagram.
    lines.append("### Winner Network Diagram")
    lines.append(f"![Winner network architecture]({png_network_rel})\n")

    # Winner genome high-level configuration.
    lines.append("### Winning Genome Configuration")
    lines.append(f"- Input nodes ({num_inputs}): `{input_keys_sorted}`")
    lines.append(f"- Hidden nodes ({num_hidden}): `{hidden_keys_sorted}`")
    lines.append(f"- Output nodes ({num_outputs}): `{output_keys_sorted}`")
    lines.append(
        f"- Total enabled connections: **{num_enabled_conns}**"
    )
    lines.append(
        f"- Pickled winner genome: `{genome_rel_path}`\n"
    )

    # ---------------------------------------------------------
    # Node table: all nodes (input / hidden / output) with type,
    # activation, and bias (where available).
    # ---------------------------------------------------------
    lines.append("#### Node Details")
    lines.append("| Node ID | Type   | Activation | Bias   |")
    lines.append("| ------- | ------ | ---------- | ------ |")

    # Build unified list of all node IDs in sorted order.
    node_ids_sorted: List[int] = (
        sorted(input_keys_sorted)
        + sorted(hidden_keys_sorted)
        + sorted(output_keys_sorted)
    )

    for node_id in node_ids_sorted:
        # Determine node type.
        if node_id in input_keys:
            node_type = "input"
        elif node_id in output_keys:
            node_type = "output"
        else:
            node_type = "hidden"

        # For hidden + output nodes, look up activation and bias from node gene.
        if node_id in winner.nodes:
            node_gene = winner.nodes[node_id]
            activation = getattr(node_gene, "activation", "unknown")
            bias = getattr(node_gene, "bias", 0.0)
            bias_str = f"{bias:.3f}"
        else:
            # Input nodes typically do not have node genes in NEAT-Python.
            activation = "N/A"
            bias_str = "N/A"

        lines.append(
            f"| {node_id} | {node_type} | {activation} | {bias_str} |"
        )

    lines.append("")  # Blank line after table

    # ---------------------------------------------------------
    # Connection table: all connections with from/to, weight,
    # and enabled status.
    # ---------------------------------------------------------
    lines.append("#### Connection Details")
    lines.append("| From | To | Weight   | Enabled |")
    lines.append("| ---- |----|----------|---------|")

    # Sort connections by (from, to) for readability.
    for (i, j) in sorted(winner.connections.keys()):
        conn = winner.connections[(i, j)]
        weight_str = f"{conn.weight:.3f}"
        enabled_str = "True" if conn.enabled else "False"
        lines.append(
            f"| {i} | {j} | {weight_str} | {enabled_str} |"
        )

    # Write the markdown file.
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Saved markdown summary to {md_path}")
