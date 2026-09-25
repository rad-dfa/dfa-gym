import re
import math
import itertools
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d import Axes3D, art3d


def parse_map(map_lines):
    """Parses the ASCII map into a 2D grid of cells."""
    grid = []
    for line in map_lines:
        cells = re.findall(r"\[(.*?)\]", line)
        if cells:
            grid.append([c.strip() for c in cells])
    return grid


def visualize(layout, figsize, cell_size=1, save_path=None, trace=None):
    map_lines = layout.splitlines()
    grid = parse_map(map_lines)
    n_rows, n_cols = len(grid), len(grid[0])

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, n_cols)
    ax.set_ylim(0, n_rows)
    ax.set_aspect("equal")
    ax.axis("off")

    for r in range(n_rows):
        for c in range(n_cols):
            content = grid[r][c]
            x, y = c, n_rows - r - 1

            # background
            ax.add_patch(patches.Rectangle(
                (x, y), cell_size, cell_size,
                facecolor="lightgray", edgecolor="white", lw=1
            ))
    agent_positions = {}
    wall_positions = {}
    for r in range(n_rows):
        for c in range(n_cols):
            content = grid[r][c]
            x, y = c, n_rows - r - 1

            if not content:
                continue

            if content == "#":  # wall
                wall_positions[(x,y)] = "dimgray"
                # ax.add_patch(patches.Rectangle(
                #     (x, y), cell_size, cell_size,
                #     facecolor="dimgray", edgecolor="black", lw=1.5
                # ))

            elif content.isupper():  # agents
                agent_positions[content] = (x + 0.5, y + 0.5)

                # ax.text(x + 0.5, y + 0.5, "8",
                #         ha="center", va="center",
                #         fontsize=14, weight="bold")

            elif content.isdigit():  # tokens
                ax.add_patch(patches.Circle(
                    (x + 0.5, y + 0.5), 0.4,
                    facecolor="gold", edgecolor="orange", lw=1.5
                ))
                ax.text(x + 0.5, y + 0.5, content,
                        ha="center", va="center",
                        fontsize=24, color="black", weight="bold")

            elif content.islower():  # sync button
                if "a" in content:
                    color = "red"
                elif "b" in content:
                    color = "green"
                elif "c" in content:
                    color = "blue"
                elif "d" in content:
                    color = "pink"
                else:
                    raise ValueError
                # color = "crimson"
                if "#" in content:
                    wall_positions[(x,y)] = color
                    # ax.add_patch(patches.Rectangle(
                    #     (x, y), cell_size, cell_size,
                    #     facecolor=color, edgecolor="black", lw=1.5,
                    #     hatch="||", hatch_linewidth=3, fill=True
                    # ))
                else:
                    ax.add_patch(patches.Rectangle(
                        (x, y), cell_size, cell_size,
                        facecolor=color, edgecolor="black", lw=1.5
                    ))

            elif "," in content:  # door like "#,a"
                parts = [p.strip() for p in content.split(",")]
                ax.add_patch(patches.Rectangle(
                    (x, y), cell_size, cell_size,
                    facecolor="firebrick", edgecolor="black", lw=1.5
                ))
                # for p in parts:
                #     if p.islower():
                #         ax.text(x + 0.5, y + 0.5, p,
                #                 ha="center", va="center",
                #                 fontsize=9, color="white")

    # if trace is not None:
    #     #TODO: Trace is a list of agent positions, where for n agents with trace length L, trace contains L many agent position entries each is a n by 2 vector giving agent positions.
    #     # Draw this trace on the map!
    if trace is not None:

        n_agents = len(agent_positions.keys())
        L = len(trace)

        # Load robot image once
        robot_img = mpimg.imread('robot.png')
        zoom = 0.05  # adjust as needed

        # Optional: add labels to track agents
        agent_labels = [str(i + 1) for i in range(n_agents)]

        # Store artists for cleanup each frame
        current_boxes = []
        current_texts = []
        current_walls = []
        current_timestep = []   # <-- NEW

        def update(frame):
            # Remove previous robot images and texts
            for ab in current_boxes:
                ab.remove()
            current_boxes.clear()

            for txt in current_texts:
                txt.remove()
            current_texts.clear()

            for wall in current_walls:
                wall.remove()
            current_walls.clear()

            for ts in current_timestep:   # <-- remove timestep text
                ts.remove()
            current_timestep.clear()

            # Add robot images and labels for this frame
            for agent_idx in range(n_agents):
                pos = trace[frame].env_state.agent_positions[agent_idx]
                x = pos[1] + 0.5
                y = n_rows - pos[0] - 0.5

                # robot image
                image_box = OffsetImage(robot_img, zoom=zoom)
                ab = AnnotationBbox(image_box, (x, y), frameon=False)
                ax.add_artist(ab)
                current_boxes.append(ab)

                # label text
                txt = ax.text(x+0.3, y + 0.3, agent_labels[agent_idx],
                              ha='center', va='bottom', color='black', weight='bold', fontsize=10)
                current_texts.append(txt)

            # Draw walls dynamically
            for i, (x, y) in enumerate(wall_positions):
                color = wall_positions[(x, y)]
                if trace[frame].env_state.is_wall_disabled[i] or color == "dimgray":
                    rect = ax.add_patch(patches.Rectangle(
                        (x, y), cell_size, cell_size,
                        facecolor=color, edgecolor="black", lw=1.5
                    ))
                else:
                    rect = ax.add_patch(patches.Rectangle(
                        (x, y), cell_size, cell_size,
                        facecolor=color, edgecolor="black", lw=1.5,
                        hatch="||", hatch_linewidth=3, fill=True
                    ))
                current_walls.append(rect)

            # Add timestep text above the grid
            ts = ax.text(n_cols / 2, n_rows + 0.5, f"Time step: {frame}",
                         ha='center', va='bottom', color='black', weight='bold', fontsize=14)
            current_timestep.append(ts)

            return current_boxes + current_texts + current_walls + current_timestep

        anim = FuncAnimation(fig, update, frames=L, interval=500, blit=False)

        if save_path:
            gif_path = save_path.replace(".pdf", ".gif")
            anim.save(gif_path, writer=PillowWriter(fps=2))

    else:

        for agent in agent_positions:
            x, y = agent_positions[agent]
            image = plt.imread('robot.png')
            image_box = OffsetImage(image, zoom=0.05)
            ab = AnnotationBbox(image_box, (x, y), frameon=False)
            ax.add_artist(ab)

        for (x, y) in wall_positions:
            color = wall_positions[(x, y)]
            ax.add_patch(patches.Rectangle(
                (x, y), cell_size, cell_size,
                facecolor=color, edgecolor="black", lw=1.5,
                hatch="||", hatch_linewidth=3, fill=True
            ))

        if save_path:
            plt.savefig(save_path, bbox_inches="tight", dpi=300)
        else:
            plt.show()
        plt.close()


def _draw_bounds(ax, low, high):
    """Draws the wireframe of a DroneEnv's bounding box and sets the axis limits to it."""
    low, high = low.tolist(), high.tolist()
    corners = list(itertools.product(*zip(low, high)))
    for a, b in itertools.combinations(corners, 2):
        if sum(u != v for u, v in zip(a, b)) == 1:
            ax.plot(*zip(a, b), color="gray", lw=0.5, ls="--")
    ax.set_xlim(low[0], high[0])
    ax.set_ylim(low[1], high[1])
    ax.set_zlim(low[2], high[2])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")


DRONE_LABEL_COLORS = {
    0: "orchid",
    1: "cornflowerblue",
    2: "yellowgreen",
    3: "sandybrown",
    4: "indianred",
}


def _cylinder_faces(cx, cy, r, z_lo, z_hi, n=20):
    """Side + cap quad/n-gon faces (each a list of (x, y, z) vertices) of a vertical cylinder."""
    ring = [(cx + r * math.cos(2 * math.pi * i / n), cy + r * math.sin(2 * math.pi * i / n)) for i in range(n)]
    faces = []
    for (x0, y0), (x1, y1) in zip(ring, ring[1:] + ring[:1]):
        faces.append([(x0, y0, z_lo), (x1, y1, z_lo), (x1, y1, z_hi), (x0, y0, z_hi)])
    faces.append([(x, y, z_lo) for x, y in ring])
    faces.append([(x, y, z_hi) for x, y in ring])
    return faces


def _box_faces(x_lo, x_hi, y_lo, y_hi, z_lo, z_hi):
    """The 6 quad faces (each a list of (x, y, z) vertices) of an axis-aligned box."""
    return [
        [(x_lo, y_lo, z_lo), (x_hi, y_lo, z_lo), (x_hi, y_hi, z_lo), (x_lo, y_hi, z_lo)],  # bottom
        [(x_lo, y_lo, z_hi), (x_hi, y_lo, z_hi), (x_hi, y_hi, z_hi), (x_lo, y_hi, z_hi)],  # top
        [(x_lo, y_lo, z_lo), (x_hi, y_lo, z_lo), (x_hi, y_lo, z_hi), (x_lo, y_lo, z_hi)],  # y = y_lo
        [(x_lo, y_hi, z_lo), (x_hi, y_hi, z_lo), (x_hi, y_hi, z_hi), (x_lo, y_hi, z_hi)],  # y = y_hi
        [(x_lo, y_lo, z_lo), (x_lo, y_hi, z_lo), (x_lo, y_hi, z_hi), (x_lo, y_lo, z_hi)],  # x = x_lo
        [(x_hi, y_lo, z_lo), (x_hi, y_hi, z_lo), (x_hi, y_hi, z_hi), (x_hi, y_lo, z_hi)],  # x = x_hi
    ]


def _draw_label_regions(ax, env, alpha=0.35):
    """Draws `env.label_f`'s regions as translucent 3D prisms (cylinder or box, 1.0 tall)."""
    for token, kind, params in env.label_regions():
        color = DRONE_LABEL_COLORS.get(token, "gray")
        if kind == "circle":
            cx, cy, r, z_lo, z_hi = (float(v) for v in params)
            faces = _cylinder_faces(cx, cy, r, z_lo, z_hi)
        else:
            x_lo, x_hi, y_lo, y_hi, z_lo, z_hi = (float(v) for v in params)
            faces = _box_faces(x_lo, x_hi, y_lo, y_hi, z_lo, z_hi)
        ax.add_collection3d(art3d.Poly3DCollection(faces, facecolor=color, edgecolor="none", alpha=alpha))

    handles = [patches.Patch(color=c, alpha=alpha, label=f"token {t}") for t, c in DRONE_LABEL_COLORS.items()]
    ax.legend(handles=handles, loc="upper left", fontsize=7, framealpha=0.6)


def visualize_drone_state(env, state, save_path=None):
    """Plots a DroneEnv state: agent positions within the environment's bounding box."""
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(projection="3d")
    _draw_bounds(ax, env.low, env.high)
    _draw_label_regions(ax, env)

    xs, ys, zs = state.positions[:, 0].tolist(), state.positions[:, 1].tolist(), state.positions[:, 2].tolist()
    ax.scatter(xs, ys, zs, s=80, c="crimson")
    for i, (x, y, z) in enumerate(zip(xs, ys, zs)):
        ax.text(x, y, z, str(i), weight="bold")

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        plt.close(fig)
    else:
        plt.show()
        plt.close(fig)


def _dfa_layout(dfa):
    """Fixed networkx layout + edge token lists for the states reachable from `dfa.start`."""
    import networkx as nx

    n_states, n_tokens = dfa.transitions.shape
    transitions = dfa.transitions.tolist()
    nodes = [s for s in range(n_states) if bool(dfa.is_reach[s])]

    edges = {}
    for s in nodes:
        for a in range(n_tokens):
            t = transitions[s][a]
            if s != t:
                edges.setdefault((s, t), []).append(a)

    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    pos = nx.shell_layout(G)
    return G, pos, edges


def _draw_dfa(ax, dfa, G, pos, edges, current, node_size=1200):
    """Draws `dfa` onto `ax` (same style as dfax.utils.visualize), highlighting state `current`.

    Edge tokens are colored like the DroneEnv label regions they correspond to.
    """
    import networkx as nx

    start = int(dfa.start)
    transitions = dfa.transitions.tolist()
    labels = dfa.labels.tolist()

    accept = [s for s in G.nodes() if labels[s]]
    reject = [s for s in G.nodes() if not labels[s] and all(t == s for t in transitions[s])]
    undecided = [s for s in G.nodes() if s not in accept and s not in reject]

    for nodelist, color in ((undecided, "white"), (accept, "#88E788"), (reject, "#FF746C")):
        nx.draw_networkx_nodes(G, pos, nodelist=nodelist, node_size=node_size, node_color=color,
                               edgecolors="black", linewidths=2, ax=ax)
    nx.draw_networkx_nodes(G, pos, nodelist=[current], node_size=node_size, node_color="none",
                           edgecolors="royalblue", linewidths=5, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=14, font_weight="bold", ax=ax)

    # Start arrow from a point to the left of the initial state.
    sx, sy = pos[start]
    ax.annotate("", xy=(sx, sy), xytext=(sx - 0.5, sy),
                arrowprops=dict(arrowstyle="-|>", lw=1.5, shrinkB=18))

    for (u, v), tokens in edges.items():
        rad = 0.25 if (v, u) in edges else 0.0
        nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], arrows=True, arrowsize=20,
                               connectionstyle=f"arc3,rad={rad}", node_size=node_size, ax=ax)

        # Token badges along the (possibly curved) edge.
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        for i, a in enumerate(tokens):
            ratio = (i + 1) / (len(tokens) + 1)
            if rad != 0:
                xc = (x0 + x1) / 2 + rad * (y1 - y0)
                yc = (y0 + y1) / 2 - rad * (x1 - x0)
                xm = (1 - ratio) ** 2 * x0 + 2 * (1 - ratio) * ratio * xc + ratio ** 2 * x1
                ym = (1 - ratio) ** 2 * y0 + 2 * (1 - ratio) * ratio * yc + ratio ** 2 * y1
            else:
                xm = x0 * (1 - ratio) + x1 * ratio
                ym = y0 * (1 - ratio) + y1 * ratio
            ax.add_patch(patches.Circle((xm, ym), 0.08, facecolor=DRONE_LABEL_COLORS.get(a, "gold"),
                                        edgecolor="black", lw=1, zorder=5))
            ax.text(xm, ym, str(a), ha="center", va="center", fontsize=11, weight="bold", zorder=6)

    ax.set_aspect("equal")
    ax.set_xlim(-1.6, 1.4)
    ax.set_ylim(-1.4, 1.4)
    ax.axis("off")


def animate_drone_trace(env, trace, save_path, fps=10, dfa=None):
    """Animates a DroneEnv trace (a list of DroneEnvState) as a GIF of agent trajectories.

    If `dfa` (a dfax.DFAx, single-agent envs only) is given, it is drawn next to the
    trajectory and advanced on `env.label_f` at every step (as DFAWrapper does, but
    without minimizing, so state ids and the layout stay fixed), with the current DFA
    state highlighted. Frame 0 is the reset state, which DFAWrapper does not label.
    """
    if dfa is not None:
        assert env.n_agents == 1, "animate_drone_trace(dfa=...) only supports single-agent traces"
        fig = plt.figure(figsize=(12, 6))
        ax = fig.add_subplot(1, 2, 1, projection="3d")
        dfa_ax = fig.add_subplot(1, 2, 2)

        G, pos, edges = _dfa_layout(dfa)
        agent = env.agents[0]
        cur = dfa
        dfa_states = [int(cur.start)]
        for s in trace[1:]:
            cur = cur.advance(env.label_f(s)[agent])
            dfa_states.append(int(cur.start))
    else:
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(projection="3d")
    _draw_bounds(ax, env.low, env.high)
    _draw_label_regions(ax, env)

    positions = [s.positions.tolist() for s in trace]
    n_agents = len(positions[0])
    scat = ax.scatter([], [], [], s=80, c="crimson")
    lines = [ax.plot([], [], [], lw=1, color=f"C{i % 10}")[0] for i in range(n_agents)]

    def update(frame):
        scat._offsets3d = tuple(zip(*positions[frame]))
        for i, line in enumerate(lines):
            hx, hy, hz = zip(*(positions[t][i] for t in range(frame + 1)))
            line.set_data(hx, hy)
            line.set_3d_properties(hz)
        ax.set_title(f"Time step: {frame}")
        if dfa is not None:
            dfa_ax.clear()
            _draw_dfa(dfa_ax, dfa, G, pos, edges, dfa_states[frame])
            dfa_ax.set_title(f"DFA state: {dfa_states[frame]}")
        return [scat, *lines]

    anim = FuncAnimation(fig, update, frames=len(trace), interval=1000 / fps, blit=False)
    anim.save(save_path, writer=PillowWriter(fps=fps))
    plt.close(fig)


if __name__ == "__main__":
    layout = """
    [ # ][ # ][ # ][ # ][ # ][   ][   ][   ][ 0 ][   ][   ][   ][ # ][ # ][ # ][ # ][ # ]
    [ # ][ 0 ][   ][ 1 ][#,c][   ][ c ][   ][ A ][   ][ a ][   ][#,a][ 0 ][   ][ 2 ][ # ]
    [ # ][   ][ 4 ][   ][#,c][   ][ c ][   ][   ][   ][ a ][   ][#,a][   ][ 8 ][   ][ # ]
    [ # ][ 3 ][   ][ 2 ][#,c][   ][ c ][   ][ B ][   ][ a ][   ][#,a][ 6 ][   ][ 4 ][ # ]
    [ # ][ # ][ # ][ # ][ # ][ 2 ][   ][   ][   ][   ][   ][ 3 ][ # ][ # ][ # ][ # ][ # ]
    [ # ][ 5 ][   ][ 6 ][#,d][   ][ d ][   ][ C ][   ][ b ][   ][#,b][ 1 ][   ][ 3 ][ # ]
    [ # ][   ][ 9 ][   ][#,d][   ][ d ][   ][   ][   ][ b ][   ][#,b][   ][ 9 ][   ][ # ]
    [ # ][ 8 ][   ][ 7 ][#,d][   ][ d ][   ][ D ][   ][ b ][   ][#,b][ 7 ][   ][ 5 ][ # ]
    [ # ][ # ][ # ][ # ][ # ][   ][   ][   ][ 1 ][   ][   ][   ][ # ][ # ][ # ][ # ][ # ]
    """
    # layout = """
    # [ # ][ # ][ # ][ # ][ # ][ # ][ # ][ # ][ # ][ # ][ # ][ # ][ # ][ # ][ # ][ # ][ # ]
    # [ # ][ 0 ][   ][ 2 ][ # ][ 0 ][   ][ 1 ][ # ][ 5 ][   ][ 6 ][ # ][ 1 ][   ][ 3 ][ # ]
    # [ # ][   ][   ][   ][ # ][   ][   ][   ][ # ][   ][   ][   ][ # ][   ][   ][   ][ # ]
    # [ # ][   ][   ][   ][ # ][   ][ B ][   ][ # ][   ][ a ][   ][ # ][   ][   ][   ][ # ]
    # [ # ][ a ][ 8 ][ A ][#,a][   ][ 4 ][   ][#,a][   ][ 9 ][   ][#,a][ D ][ 9 ][ a ][ # ]
    # [ # ][   ][   ][   ][ # ][   ][ a ][   ][ # ][   ][ C ][   ][ # ][   ][   ][   ][ # ]
    # [ # ][   ][   ][   ][ # ][   ][   ][   ][ # ][   ][   ][   ][ # ][   ][   ][   ][ # ]
    # [ # ][ 6 ][   ][ 4 ][ # ][ 3 ][   ][ 2 ][ # ][ 8 ][   ][ 7 ][ # ][ 7 ][   ][ 5 ][ # ]
    # [ # ][ # ][ # ][ # ][ # ][ # ][ # ][ # ][ # ][ # ][ # ][ # ][ # ][ # ][ # ][ # ][ # ]
    # """
    # layout = """
    # [ 0 ][   ][   ][   ][ # ][ # ][ # ][ # ][ # ]
    # [   ][   ][ a ][   ][#,a][ 0 ][   ][ 2 ][ # ]
    # [ A ][   ][ a ][   ][#,a][   ][ 8 ][   ][ # ]
    # [   ][   ][ a ][   ][#,a][ 6 ][   ][ 4 ][ # ]
    # [ 1 ][   ][   ][ 3 ][ # ][ # ][ # ][ # ][ # ]
    # [   ][   ][ b ][   ][#,b][ 1 ][   ][ 3 ][ # ]
    # [ B ][   ][ b ][   ][#,b][   ][ 9 ][   ][ # ]
    # [   ][   ][ b ][   ][#,b][ 7 ][   ][ 5 ][ # ]
    # [ 2 ][   ][   ][   ][ # ][ # ][ # ][ # ][ # ]
    # """
    # layout = """
    # [ # ][ # ][ # ][ # ][ # ][ # ][ # ][ # ][ # ]
    # [ # ][ 0 ][   ][ 2 ][ # ][ 1 ][   ][ 3 ][ # ]
    # [ # ][   ][   ][   ][ # ][   ][   ][   ][ # ]
    # [ # ][   ][   ][   ][ # ][   ][   ][   ][ # ]
    # [ # ][ a ][ 8 ][ A ][#,a][ B ][ 9 ][ a ][ # ]
    # [ # ][   ][   ][   ][ # ][   ][   ][   ][ # ]
    # [ # ][   ][   ][   ][ # ][   ][   ][   ][ # ]
    # [ # ][ 6 ][   ][ 4 ][ # ][ 7 ][   ][ 5 ][ # ]
    # [ # ][ # ][ # ][ # ][ # ][ # ][ # ][ # ][ # ]
    # """

    # visualize(layout, figsize=(17,9), save_path="maps/4buttons_4agents.pdf")
    visualize(layout, figsize=(17,9))
