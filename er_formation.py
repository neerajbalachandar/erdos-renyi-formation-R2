"""
Problem 1: Formation-control name formation with an Erdos-Renyi graph.

The implementation follows the formation-control idea from the lecture:

    z_i = p_i - r_i

    dot(z_i) = sum_j a_ij (z_j - z_i)

where p_i is the actual agent position and r_i is the desired position

assigned to agent i for the current letter. Because the graph is connected

and the centroid is fixed to the same location for every target formation,

z -> 0, so p_i -> r_i.

To run: pip install numpy matplotlib networkx pillow

"""

import numpy as np

import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter

import networkx as nx

from pathlib import Path


NAME = "NEERAJ"

N = 20

P_EDGE = 0.25        # ER edge probability

SEED = 7

DT = 0.04

FORMATION_TOL = 0.04

MAX_STEPS_PER_LETTER = 500

MIN_STEPS_PER_LETTER = 15

HOLD_SECONDS = 0.50

FPS = 18

FORMATION_GAIN = 2.0

FORMATION_CENTER = np.array([0.0, 0.0])


LETTER_STROKES = {

    "A": [
        [(-0.9, -1.2), (0.0, 1.2), (0.9, -1.2)],
        [(-0.48, -0.15), (0.48, -0.15)],
    ],

    "E": [
        [(-0.8, -1.2), (-0.8, 1.2)],
        [(-0.8, 1.2), (0.8, 1.2)],
        [(-0.8, 0.0), (0.55, 0.0)],
        [(-0.8, -1.2), (0.8, -1.2)],
    ],

    "J": [
        [(0.65, 1.2), (0.65, -0.7)],
        [(0.65, -0.7), (0.35, -1.15)],
        [(0.35, -1.15), (-0.35, -1.15)],
        [(-0.35, -1.15), (-0.65, -0.7)],
    ],

    "N": [
        [(-0.8, -1.2), (-0.8, 1.2)],
        [(-0.8, 1.2), (0.8, -1.2)],
        [(0.8, -1.2), (0.8, 1.2)],
    ],

    "R": [
        [(-0.8, -1.2), (-0.8, 1.2)],
        [(-0.8, 1.2), (0.15, 1.2)],
        [(0.15, 1.2), (0.65, 0.75)],
        [(0.65, 0.75), (0.15, 0.2)],
        [(0.15, 0.2), (-0.8, 0.2)],
        [(0.05, 0.2), (0.8, -1.2)],
    ],

    "V": [
        [(-0.9, 1.2), (0.0, -1.2), (0.9, 1.2)],
    ],

}


def segment(a, b, n):

    """Return n equally spaced points on the line segment a -> b."""

    a = np.asarray(a, dtype=float)

    b = np.asarray(b, dtype=float)

    t = np.linspace(0.0, 1.0, n)

    return (1.0 - t[:, None]) * a + t[:, None] * b


def sample_stroke(stroke, n):

    """

    Sample n points along a polyline approximately uniformly in arc length.

    """

    stroke = np.asarray(stroke, dtype=float)

    if len(stroke) == 2:

        return segment(stroke[0], stroke[1], n)

    seg_len = np.array([

        np.linalg.norm(stroke[k + 1] - stroke[k])

        for k in range(len(stroke) - 1)

    ])

    cum = np.concatenate(([0.0], np.cumsum(seg_len)))

    total = cum[-1]

    s_query = np.linspace(0.0, total, n)

    pts = np.zeros((n, 2))

    for i, s in enumerate(s_query):

        k = np.searchsorted(cum, s, side="right") - 1

        k = min(k, len(seg_len) - 1)

        local = 0.0 if seg_len[k] == 0 else (s - cum[k]) / seg_len[k]

        pts[i] = (1 - local) * stroke[k] + local * stroke[k + 1]

    return pts


def dense_segment(a, b, n=200):

    a = np.asarray(a, dtype=float)

    b = np.asarray(b, dtype=float)

    t = np.linspace(0.0, 1.0, n, endpoint=False)

    return (1.0 - t[:, None]) * a + t[:, None] * b


def dense_stroke(stroke, samples_per_segment=200):

    pieces = []

    for k in range(len(stroke) - 1):

        pieces.append(

            dense_segment(

                stroke[k],

                stroke[k + 1],

                samples_per_segment,

            )

        )

    pieces.append(

        np.asarray(stroke[-1], dtype=float)[None, :]

    )

    return np.vstack(pieces)


def remove_duplicate_points(points, decimals=8):

    rounded = np.round(points, decimals=decimals)

    _, indices = np.unique(

        rounded,

        axis=0,

        return_index=True,

    )

    return points[np.sort(indices)]


def farthest_point_sampling(points, n, mandatory_points=None):

    points = np.asarray(points, dtype=float)

    mandatory_points = np.asarray(

        mandatory_points if mandatory_points is not None else [],

        dtype=float

    )

    if mandatory_points.size == 0:

        mandatory_points = np.empty((0, 2))

    else:

        mandatory_points = remove_duplicate_points(

            mandatory_points

        )

    selected = []

    if len(mandatory_points) > 0:

        selected.extend(

            [p.copy() for p in mandatory_points]

        )

    if len(selected) > n:

        selected_array = np.asarray(selected)

        selected_indices = [0]

        min_dist_sq = np.full(

            len(selected_array),

            np.inf,

        )

        for _ in range(1, n):

            newest = selected_array[

                selected_indices[-1]

            ]

            dist_sq = np.sum(

                (selected_array - newest) ** 2,

                axis=1,

            )

            min_dist_sq = np.minimum(

                min_dist_sq,

                dist_sq,

            )

            min_dist_sq[selected_indices] = -1.0

            selected_indices.append(

                int(np.argmax(min_dist_sq))

            )

        return selected_array[selected_indices]

    candidates = points.copy()

    if len(selected) == 0:

        centre = candidates.mean(axis=0)

        first = np.argmin(

            np.linalg.norm(

                candidates - centre,

                axis=1,

            )

        )

        selected.append(

            candidates[first].copy()

        )

        candidates = np.delete(

            candidates,

            first,

            axis=0,

        )

    else:

        selected_array = np.asarray(selected)

        distances_sq = np.min(

            np.sum(

                (

                    candidates[:, None, :]

                    - selected_array[None, :, :]

                ) ** 2,

                axis=2,

            ),

            axis=1,

        )

        while (

            len(selected) < n

            and len(candidates) > 0

        ):

            next_index = int(

                np.argmax(distances_sq)

            )

            selected.append(

                candidates[next_index].copy()

            )

            candidates = np.delete(

                candidates,

                next_index,

                axis=0,

            )

            if len(candidates) == 0:

                break

            selected_array = np.asarray(selected)

            distances_sq = np.min(

                np.sum(

                    (

                        candidates[:, None, :]

                        - selected_array[None, :, :]

                    ) ** 2,

                    axis=2,

                ),

                axis=1,

            )

    while (

        len(selected) < n

        and len(candidates) > 0

    ):

        selected_array = np.asarray(selected)

        distances_sq = np.min(

            np.sum(

                (

                    candidates[:, None, :]

                    - selected_array[None, :, :]

                ) ** 2,

                axis=2,

            ),

            axis=1,

        )

        next_index = int(

            np.argmax(distances_sq)

        )

        selected.append(

            candidates[next_index].copy()

        )

        candidates = np.delete(

            candidates,

            next_index,

            axis=0,

        )

    return np.asarray(selected)


def make_letter(letter, n=N, scale=2.0):

    """

    Create exactly n target points for the requested capital letter.

    """

    strokes = LETTER_STROKES[letter]

    dense_points = []

    important_points = []

    for stroke in strokes:

        dense_points.append(

            dense_stroke(stroke)

        )

        important_points.extend(

            np.asarray(stroke, dtype=float)

        )

    dense_points = remove_duplicate_points(

        np.vstack(dense_points)

    )

    important_points = remove_duplicate_points(

        np.vstack(important_points)

    )

    points = farthest_point_sampling(

        dense_points,

        n,

        mandatory_points=important_points,

    )

    # Normalize and centre the letter.

    points -= points.mean(axis=0)

    extent = max(

        np.ptp(points[:, 0]),

        np.ptp(points[:, 1]),

    )

    points *= scale / extent

    return points


# Generate connected ER graph

def generate_connected_er_graph(n, p, seed):

    rng = np.random.default_rng(seed)

    attempts = 0

    while True:

        attempts += 1

        graph_seed = int(

            rng.integers(

                0,

                2**32 - 1,

            )

        )

        G = nx.erdos_renyi_graph(

            n,

            p,

            seed=graph_seed,

        )

        if nx.is_connected(G):

            A = nx.to_numpy_array(

                G,

                dtype=float,

            )

            return G, A, attempts


G, A, attempts = generate_connected_er_graph(

    N,

    P_EDGE,

    SEED

)

print("ER Graph")

print(f"N = {N}")

print(f"p = {P_EDGE}")

print(f"Edges = {G.number_of_edges()}")

print(

    f"Average degree = "

    f"{2 * G.number_of_edges() / N:.2f}"

)

print(

    f"Connected = "

    f"{nx.is_connected(G)}"

)

print(

    f"Attempts needed = {attempts}"

)


# Initial Positions

rng = np.random.default_rng(

    SEED + 1

)

x = rng.uniform(

    -5.0,

    5.0,

    size=(N, 2)

)

x += (

    FORMATION_CENTER

    - x.mean(axis=0)

)

initial_positions = x.copy()


# Build Target formation

letters = list(NAME)

letter_targets = []

for letter in letters:

    target = make_letter(

        letter,

        n=N,

        scale=2.0

    )

    target += (

        FORMATION_CENTER

        - target.mean(axis=0)

    )

    letter_targets.append(target)


# Formation-control law

degree_matrix = np.diag(

    A.sum(axis=1)

)

L = degree_matrix - A


def formation_control(x, r):

    """

    Formation controller:

        u_i = sum_j a_ij [ (x_j - x_i) - (r_j - r_i) ]

    First term:

        agreement of transformed coordinates z_i = x_i - r_i.

    """

    return (

        -FORMATION_GAIN

        * L

        @ (x - r)

    )


# Sequential motion

trajectory = []

letter_index_per_frame = []

for letter_index, target in enumerate(

    letter_targets

):

    for step in range(

        MAX_STEPS_PER_LETTER

    ):

        error = np.mean(

            np.linalg.norm(

                x - target,

                axis=1

            )

        )

        if (

            error <= FORMATION_TOL

            and step >= MIN_STEPS_PER_LETTER

        ):

            break

        u = formation_control(

            x,

            target

        )

        # Single-integrator discrete-time dynamics:

        # x_i[k+1] = x_i[k] + dt * u_i[k]

        x = x + DT * u

        trajectory.append(

            x.copy()

        )

        letter_index_per_frame.append(

            letter_index

        )

    # Hold the converged letter for the animation.

    hold_frames = max(

        1,

        int(

            HOLD_SECONDS * FPS

        )

    )

    for _ in range(hold_frames):

        trajectory.append(

            x.copy()

        )

        letter_index_per_frame.append(

            letter_index

        )


trajectory = np.asarray(

    trajectory

)


# Convergence error for every letter

print("\n======= FORMATION ERRORS =======")

for k, letter in enumerate(letters):

    target = letter_targets[k]

    letter_frames = [

        i

        for i, letter_index

        in enumerate(

            letter_index_per_frame

        )

        if letter_index == k

    ]

    final_positions = trajectory[

        letter_frames[-1]

    ]

    agent_errors = np.linalg.norm(

        final_positions - target,

        axis=1

    )

    mean_error = agent_errors.mean()

    max_error = agent_errors.max()

    actual_centered = (

        final_positions

        - final_positions.mean(axis=0)

    )

    target_centered = (

        target

        - target.mean(axis=0)

    )

    shape_error = np.mean(

        np.linalg.norm(

            actual_centered - target_centered,

            axis=1

        )

    )

    print(

        f"{letter}: mean error = "

        f"{mean_error:.4e}, "

        f"max error = "

        f"{max_error:.4e}, "

        f"shape error = "

        f"{shape_error:.4e}"

    )


# Animation

fig, ax = plt.subplots(

    figsize=(12, 6)

)

all_points = np.concatenate(

    [

        initial_positions[None, :, :],

        trajectory

    ],

    axis=0

)

xmin = all_points[:, :, 0].min() - 1.0

xmax = all_points[:, :, 0].max() + 1.0

ymin = all_points[:, :, 1].min() - 1.0

ymax = all_points[:, :, 1].max() + 1.0

ax.set_xlim(

    xmin,

    xmax

)

ax.set_ylim(

    ymin,

    ymax

)

ax.set_aspect(

    "equal",

    adjustable="box"

)

ax.set_xlabel("x")

ax.set_ylabel("y")

ax.set_title(

    "Distributed Formation Control"

)

scatter = ax.scatter(

    [],

    [],

    s=60

)

# Draw communication graph.

edge_artists = []

for i, j in G.edges():

    line, = ax.plot(

        [],

        [],

        linewidth=0.7,

        alpha=0.30

    )

    edge_artists.append(

        (line, i, j)

    )

title_text = ax.text(

    0.02,

    0.95,

    "",

    transform=ax.transAxes,

    fontsize=13,

    va="top"

)


def update(frame):

    positions = trajectory[frame]

    letter = letters[

        letter_index_per_frame[frame]

    ]

    scatter.set_offsets(

        positions

    )

    for line, i, j in edge_artists:

        line.set_data(

            [

                positions[i, 0],

                positions[j, 0]

            ],

            [

                positions[i, 1],

                positions[j, 1]

            ]

        )

    title_text.set_text(

        f"Name = {NAME}    "

        f"Current letter = {letter}"

    )

    return [

        scatter,

        title_text,

        *(

            line

            for line, _, _ in edge_artists

        ),

    ]


animation = FuncAnimation(

    fig,

    update,

    frames=len(trajectory),

    interval=1000 / FPS,

    blit=False

)

output_video = Path(

    "name_formation.mp4"

)

output_gif = Path(

    "name_formation.gif"

)


try:

    animation.save(

        output_video,

        writer=FFMpegWriter(

            fps=FPS,

            bitrate=1800

        )

    )

    print(

        f"\nSaved MP4: "

        f"{output_video.resolve()}"

    )

except Exception as exc:

    print(

        "\nFFmpeg was not available, "

        "so MP4 was skipped."

    )

    print(

        "Reason:",

        exc

    )

    animation.save(

        output_gif,

        writer=PillowWriter(

            fps=FPS

        )

    )

    print(

        f"Saved GIF: "

        f"{output_gif.resolve()}"

    )

plt.close(fig)


# Graph visualization at time instant?

plt.figure(

    figsize=(7, 7)

)

layout = nx.spring_layout(

    G,

    seed=SEED

)

nx.draw_networkx(

    G,

    pos=layout,

    node_size=250,

    with_labels=True,

    font_size=7

)

plt.title(

    f"Connected Erdős-Rényi Graph "

    f"(N={N}, p={P_EDGE})"

)

plt.axis("off")

plt.tight_layout()

plt.savefig(

    "er_graph.png",

    dpi=200

)

plt.close()

print(

    "Saved graph image: er_graph.png"

)