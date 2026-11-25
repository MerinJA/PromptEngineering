"""Simulation 2D pédagogique : baseline réactive contre DOA-PVO."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import csv
import math
import random
import secrets
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np


DT = 0.10
DRONE_RADIUS = 0.22
OBSTACLE_RADIUS = 0.28
SAFETY_MARGIN = 0.12
MAX_SPEED = 1.45
MAX_ACCEL = 3.0
GOAL_TOLERANCE = 0.30
PREDICTION_TIMES = np.arange(0.20, 2.01, 0.20)
POSITION_NOISE_STD = 0.03
METHODS = ("Reactive", "DOA-PVO")


@dataclass
class MovingObstacle:
    """Obstacle à déplacement horizontal avec oscillation verticale non linéaire."""

    initial: np.ndarray
    velocity: np.ndarray
    amplitude: float = 0.0
    frequency: float = 0.0
    phase: float = 0.0

    def position(self, t: float) -> np.ndarray:
        p = self.initial + self.velocity * t
        p[1] += self.amplitude * math.sin(self.frequency * t + self.phase)
        return p

@dataclass
class Scenario:
    name: str
    start: np.ndarray
    goal: np.ndarray
    obstacles: list[MovingObstacle]
    sensor_seed: int = 0


SCENARIOS = [
    Scenario(
        "Croisement latéral",
        np.array([0.0, 0.0]), np.array([10.0, 0.0]),
        [MovingObstacle(np.array([5.2, -2.1]), np.array([0.0, 0.95]), 0.32, 2.1, 0.2)], 101,
    ),
    Scenario(
        "Trajectoire oscillante",
        np.array([0.0, 0.0]), np.array([10.0, 0.0]),
        [MovingObstacle(np.array([5.4, 0.15]), np.array([0.0, 0.0]), 1.10, 1.55, 0.7)], 102,
    ),
    Scenario(
        "Double passage",
        np.array([0.0, 0.0]), np.array([10.0, 0.0]),
        [
            MovingObstacle(np.array([4.4, -2.3]), np.array([0.0, 0.92]), 0.22, 1.8, 0.0),
            MovingObstacle(np.array([6.8, 2.1]), np.array([0.0, -0.82]), 0.28, 1.7, 1.3),
        ], 103,
    ),
]


def random_scenarios(count: int = 60, seed: int | None = None) -> list[Scenario]:
    """Construit des rencontres dynamiques comparables et reproductibles.

    Chaque obstacle traverse approximativement le couloir direct du drone ; les
    scénarios ne sont donc pas artificiellement faciles.
    """
    rng = random.Random(seed)
    scenarios: list[Scenario] = []
    for index in range(count):
        obstacles: list[MovingObstacle] = []
        for _ in range(rng.choice([1, 1, 2])):
            x = rng.uniform(3.2, 7.7)
            side = rng.choice([-1.0, 1.0])
            y = side * rng.uniform(0.75, 2.30)
            crossing_time = x / rng.uniform(0.85, 1.20)
            vertical_speed = -y / crossing_time
            obstacles.append(
                MovingObstacle(
                    initial=np.array([x, y]),
                    velocity=np.array([rng.uniform(-0.08, 0.08), vertical_speed]),
                    amplitude=rng.uniform(0.08, 0.42),
                    frequency=rng.uniform(1.0, 2.3),
                    phase=rng.uniform(0, 2 * math.pi),
                )
            )
        scenarios.append(
            Scenario(
                f"Random {index + 1:02d}", np.array([0.0, 0.0]), np.array([10.0, 0.0]),
                obstacles, rng.randrange(2**32),
            )
        )
    return scenarios


def candidate_velocities(current: np.ndarray, goal: np.ndarray) -> list[np.ndarray]:
    """Échantillon de vitesses avec une densité accrue vers le but."""
    direction = goal - current
    direction /= max(np.linalg.norm(direction), 1e-9)
    preferred_angle = math.atan2(direction[1], direction[0])
    candidates = [np.zeros(2)]
    for speed in np.linspace(0.05, MAX_SPEED, 15):
        for offset in np.linspace(-math.pi, math.pi, 25, endpoint=False):
            angle = preferred_angle + offset
            candidates.append(speed * np.array([math.cos(angle), math.sin(angle)]))
    return candidates


def observed_position(scenario: Scenario, obstacle: MovingObstacle, obstacle_index: int, t: float) -> np.ndarray:
    """Mesure bruitée et reproductible de la position, sans accès à la trajectoire future."""
    step = max(0, int(round(t / DT)))
    generator = np.random.default_rng(np.random.SeedSequence([scenario.sensor_seed, obstacle_index, step]))
    return obstacle.position(t) + generator.normal(0.0, POSITION_NOISE_STD, size=2)


def obstacle_observations(scenario: Scenario, t: float) -> list[tuple[np.ndarray, np.ndarray]]:
    """Produit une unique observation par obstacle et par instant de contrôle."""
    observations = []
    for obstacle_index, obstacle in enumerate(scenario.obstacles):
        current_measurement = observed_position(scenario, obstacle, obstacle_index, t)
        previous_measurement = observed_position(scenario, obstacle, obstacle_index, max(0.0, t - DT))
        estimated_velocity = (current_measurement - previous_measurement) / max(t - max(0.0, t - DT), DT)
        observations.append((current_measurement, estimated_velocity))
    return observations


def is_safe(
    position: np.ndarray,
    velocity: np.ndarray,
    observations: list[tuple[np.ndarray, np.ndarray]],
    predictive: bool,
) -> tuple[bool, float]:
    """Évalue une vitesse à partir de mesures présentes, et non de la loi vraie du mouvement."""
    horizon = PREDICTION_TIMES if predictive else np.array([0.45])
    clearance = float("inf")
    for current_measurement, estimated_velocity in observations:
        for tau in horizon:
            drone_future = position + velocity * tau
            # La baseline ne voit que la position présente. DOA-PVO extrapole la
            # vitesse estimée par différence finie sur deux mesures bruitées.
            obstacle_future = current_measurement + estimated_velocity * tau if predictive else current_measurement
            distance = np.linalg.norm(drone_future - obstacle_future) - (DRONE_RADIUS + OBSTACLE_RADIUS)
            clearance = min(clearance, distance)
            if distance < SAFETY_MARGIN:
                return False, clearance
    return True, clearance if math.isfinite(clearance) else 999.0


def choose_velocity(
    position: np.ndarray,
    previous: np.ndarray,
    scenario: Scenario,
    t: float,
    predictive: bool,
    observations: list[tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    best, best_cost = np.zeros(2), float("inf")
    for candidate in candidate_velocities(position, scenario.goal):
        if np.linalg.norm(candidate - previous) / DT > MAX_ACCEL:
            continue
        safe, clearance = is_safe(position, candidate, observations, predictive)
        if not safe:
            continue
        next_position = position + candidate * DT
        goal_cost = np.linalg.norm(scenario.goal - next_position)
        turn_cost = 0.02 * np.linalg.norm(candidate - previous)
        # Favorise une marge utile, sans rendre la trajectoire inutilement longue.
        clearance_bonus = 0.12 * min(clearance, 2.0)
        cost = goal_cost + turn_cost - clearance_bonus
        if cost < best_cost:
            best, best_cost = candidate, cost
    return best


def simulate(scenario: Scenario, method: str, max_time: float = 18.0) -> dict:
    predictive = method == "DOA-PVO"
    position = scenario.start.astype(float).copy()
    velocity = np.zeros(2)
    path, clearances = [position.copy()], []
    collided, reached = False, False
    steps = int(max_time / DT)
    for step in range(steps):
        t = step * DT
        observations = obstacle_observations(scenario, t)
        velocity = choose_velocity(position, velocity, scenario, t, predictive, observations)
        position += velocity * DT
        min_clearance = min(
            np.linalg.norm(position - obstacle.position(t + DT)) - (DRONE_RADIUS + OBSTACLE_RADIUS)
            for obstacle in scenario.obstacles
        )
        clearances.append(min_clearance)
        path.append(position.copy())
        if min_clearance < 0:
            collided = True
            break
        if np.linalg.norm(scenario.goal - position) <= GOAL_TOLERANCE:
            reached = True
            break
    path = np.asarray(path)
    travel_distance = float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())
    return {
        "scenario": scenario.name,
        "method": method,
        "success": reached and not collided,
        "collision": collided,
        "time_s": len(path) * DT,
        "distance_m": travel_distance,
        "min_clearance_m": float(min(clearances)) if clearances else float("nan"),
        "path": path,
    }


def plot_trajectories(results: list[dict], output: Path) -> None:
    fig, axes = plt.subplots(1, len(SCENARIOS), figsize=(15, 4.4), sharey=True)
    styles = {"Reactive": ("#d55e00", "--"), "DOA-PVO": ("#0072b2", "-")}
    for axis, scenario in zip(axes, SCENARIOS):
        time_points = np.arange(0, 18.01, 0.1)
        for obstacle in scenario.obstacles:
            points = np.array([obstacle.position(t) for t in time_points])
            axis.plot(points[:, 0], points[:, 1], color="#777777", alpha=0.45, linewidth=1.0)
            axis.add_patch(plt.Circle(obstacle.position(0), OBSTACLE_RADIUS, color="#555555", alpha=0.55))
        for result in [r for r in results if r["scenario"] == scenario.name]:
            color, line = styles[result["method"]]
            axis.plot(result["path"][:, 0], result["path"][:, 1], line, color=color, linewidth=2.2, label=result["method"])
        axis.scatter(*scenario.start, c="#009e73", s=48, zorder=5, label="Départ")
        axis.scatter(*scenario.goal, c="#cc79a7", marker="*", s=115, zorder=5, label="But")
        axis.set_title(scenario.name)
        axis.set_xlabel("x (m)")
        axis.set_xlim(-0.5, 10.7)
        axis.set_ylim(-4.0, 4.0)
        axis.grid(alpha=0.22)
    axes[0].set_ylabel("y (m)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
    fig.suptitle("Anticiper les obstacles mobiles : réactif vs DOA-PVO", y=1.02, fontsize=14)
    fig.tight_layout(rect=(0, 0.08, 1, 0.96))
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_metrics(results: list[dict], output: Path) -> None:
    labels = [r["scenario"] for r in results if r["method"] == "Reactive"]
    metrics = [("Distance minimale (m)", "min_clearance_m"), ("Durée (s)", "time_s"), ("Distance parcourue (m)", "distance_m")]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    x = np.arange(len(labels))
    for axis, (title, key) in zip(axes, metrics):
        reactive = [next(r[key] for r in results if r["scenario"] == name and r["method"] == "Reactive") for name in labels]
        pvo = [next(r[key] for r in results if r["scenario"] == name and r["method"] == "DOA-PVO") for name in labels]
        axis.bar(x - 0.19, reactive, 0.38, label="Reactive", color="#d55e00")
        axis.bar(x + 0.19, pvo, 0.38, label="DOA-PVO", color="#0072b2")
        axis.set_title(title)
        axis.set_xticks(x, ["S1", "S2", "S3"])
        axis.grid(axis="y", alpha=0.22)
    axes[0].legend(frameon=False)
    fig.suptitle("Mesures calculées sur les trois scénarios", y=1.02, fontsize=14)
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_random_trajectories(results: list[dict], scenarios: list[Scenario], output: Path) -> None:
    """Montre les six premiers scénarios aléatoires lisibles où les deux méthodes réussissent."""
    paired = []
    for scenario in scenarios:
        reactive = next(r for r in results if r["scenario"] == scenario.name and r["method"] == "Reactive")
        pvo = next(r for r in results if r["scenario"] == scenario.name and r["method"] == "DOA-PVO")
        # Les exemples pédagogiques doivent rester lisibles dans le même cadre,
        # sans masquer les échecs dans le résumé statistique global.
        if reactive["success"] and pvo["success"] and max(np.abs(reactive["path"][:, 1])) < 2.8 and max(np.abs(pvo["path"][:, 1])) < 2.8:
            paired.append((pvo["min_clearance_m"] - reactive["min_clearance_m"], scenario, reactive, pvo))
    selected = paired[:6]
    fig, axes = plt.subplots(2, 3, figsize=(13.6, 7.4), sharex=True, sharey=True)
    for index, (_, scenario, reactive, pvo) in enumerate(selected):
        axis = axes.flat[index]
        for obstacle in scenario.obstacles:
            times = np.arange(0, 10.01, DT)
            points = np.array([obstacle.position(t) for t in times])
            axis.plot(points[:, 0], points[:, 1], color="#808080", alpha=0.42, linewidth=1.1)
            axis.add_patch(plt.Circle(obstacle.position(0), OBSTACLE_RADIUS, color="#555555", alpha=0.55))
        axis.plot(reactive["path"][:, 0], reactive["path"][:, 1], "--", color="#d55e00", linewidth=2, label="Reactive")
        axis.plot(pvo["path"][:, 0], pvo["path"][:, 1], color="#0072b2", linewidth=2.2, label="DOA-PVO")
        axis.scatter(*scenario.start, c="#009e73", s=34, zorder=5)
        axis.scatter(*scenario.goal, c="#cc79a7", marker="*", s=95, zorder=5)
        axis.set_title(f"Scenario {index + 1}: clearance +{pvo['min_clearance_m'] - reactive['min_clearance_m']:.2f} m")
        axis.set_xlim(-0.4, 10.55)
        axis.set_ylim(-3.4, 3.4)
        axis.grid(alpha=0.2)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    fig.text(0.5, 0.06, "x position (m)", ha="center")
    fig.text(0.03, 0.5, "y position (m)", va="center", rotation="vertical")
    fig.suptitle("First six readable random obstacle encounters", y=0.985, fontsize=15)
    fig.tight_layout(rect=(0.04, 0.10, 1, 0.95))
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_random_summary(results: list[dict], output: Path, seed: int) -> None:
    methods = METHODS
    clearance = [[r["min_clearance_m"] for r in results if r["method"] == method] for method in methods]
    distance = [[r["distance_m"] for r in results if r["method"] == method] for method in methods]
    success = [100 * np.mean([r["success"] for r in results if r["method"] == method]) for method in methods]
    colors = ["#d55e00", "#0072b2"]
    fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.6))
    for axis, values, title, ylabel in [
        (axes[0], clearance, "Safety margin distribution", "Minimum clearance (m)"),
        (axes[1], distance, "Route-length distribution", "Travel distance (m)"),
    ]:
        boxes = axis.boxplot(values, tick_labels=methods, patch_artist=True, showmeans=True)
        for box, color in zip(boxes["boxes"], colors):
            box.set_facecolor(color)
            box.set_alpha(0.75)
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=0.22)
    bars = axes[2].bar(methods, success, color=colors, width=0.55)
    axes[2].set_ylim(0, 105)
    axes[2].set_ylabel("Successful runs (%)")
    axes[2].set_title("Goal-reaching success")
    axes[2].grid(axis="y", alpha=0.22)
    for bar, value in zip(bars, success):
        axes[2].text(bar.get_x() + bar.get_width() / 2, value + 2, f"{value:.0f}%", ha="center", va="bottom")
    fig.suptitle(f"Random dynamic-obstacle scenarios (seed: {seed})", y=1.02, fontsize=15)
    fig.tight_layout()
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare a reactive baseline and DOA-PVO in dynamic-obstacle simulations.")
    parser.add_argument("--seed", type=int, help="Optional seed to reproduce a specific random campaign.")
    parser.add_argument("--runs", type=int, default=60, help="Number of random scenarios (default: 60).")
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs must be at least 1")
    # A new seed is drawn on each normal run. It is written to disk and printed,
    # so any interesting outcome can still be reproduced exactly with --seed.
    seed = args.seed if args.seed is not None else secrets.randbelow(2**32)
    output = Path(__file__).parent / "outputs"
    output.mkdir(exist_ok=True)
    results = [simulate(scenario, method) for scenario in SCENARIOS for method in METHODS]
    with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["scenario", "method", "success", "collision", "time_s", "distance_m", "min_clearance_m"])
        writer.writeheader()
        for result in results:
            writer.writerow({key: result[key] for key in writer.fieldnames})
    plot_trajectories(results, output / "trajectories.png")
    plot_metrics(results, output / "metrics.png")
    random_cases = random_scenarios(args.runs, seed)
    random_results = [simulate(scenario, method) for scenario in random_cases for method in METHODS]
    with (output / "random_metrics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["scenario", "method", "success", "collision", "time_s", "distance_m", "min_clearance_m"])
        writer.writeheader()
        for result in random_results:
            writer.writerow({key: result[key] for key in writer.fieldnames})
    summary_lines = [
        "=" * 72,
        "RANDOM DYNAMIC-OBSTACLE CAMPAIGN",
        f"seed={seed}",
        f"iterations={args.runs}",
        "",
    ]
    for method in METHODS:
        method_results = [result for result in random_results if result["method"] == method]
        success_rate = 100 * sum(result["success"] for result in method_results) / len(method_results)
        mean_clearance = float(np.mean([result["min_clearance_m"] for result in method_results]))
        mean_distance = float(np.mean([result["distance_m"] for result in method_results]))
        summary_lines.extend([
            f"[{method}]",
            f"success_rate_percent={success_rate:.2f}",
            f"mean_minimum_clearance_m={mean_clearance:.3f}",
            f"mean_route_length_m={mean_distance:.3f}",
            "",
        ])
    summary_lines.append(f"rerun: python run_experiments.py --seed {seed} --runs {args.runs}")
    summary_lines.append("=" * 72)
    summary_text = "\n".join(summary_lines) + "\n"
    history_path = output / "experiment_history.log"
    run_timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
    history_block = f"timestamp_local={run_timestamp}\n" + summary_text
    with history_path.open("a", encoding="utf-8") as file:
        file.write(history_block)
    plot_random_trajectories(random_results, random_cases, output / "classroom_random_trajectories.png")
    plot_random_summary(random_results, output / "classroom_random_summary.png", seed)
    print(f"Random campaign: {args.runs} scenarios, seed={seed}")
    for result in results:
        print(
            f'{result["scenario"]:24} | {result["method"]:7} | succès={result["success"]} '
            f'| marge min={result["min_clearance_m"]:.2f} m | durée={result["time_s"]:.1f} s'
        )


if __name__ == "__main__":
    print("\nStarting...\n\n")
    main()
    print("\n\nEnd of the experiment.\nPlease check 'outputs'\n")
