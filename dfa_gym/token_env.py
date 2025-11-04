import jax
import chex
import numpy as np
import jax.numpy as jnp
from flax import struct
from enum import IntEnum
from functools import partial
from typing import Tuple, Dict
from dfa_gym import spaces
from dfa_gym.env import MultiAgentEnv, State


class Action(IntEnum):
    DOWN  = 0
    RIGHT = 1
    UP    = 2
    LEFT  = 3
    NOOP  = 4

ACTION_MAP = jnp.array([
    [ 1,  0], # DOWN
    [ 0,  1], # RIGHT
    [-1,  0], # UP
    [ 0, -1], # LEFT
    [ 0,  0], # NOOP
])

@struct.dataclass
class TokenEnvState(State):
    agent_positions: jax.Array
    token_positions: jax.Array
    wall_positions: jax.Array
    is_wall_disabled: jax.Array
    button_positions: jax.Array
    is_alive: jax.Array
    time: int

class TokenEnv(MultiAgentEnv):

    def __init__(
        self,
        n_agents: int = 3,
        n_tokens: int = 10,
        n_token_repeat: int = 2,
        grid_shape: Tuple[int, int] = (7, 7),
        fixed_map_seed: int | None = None,
        max_steps_in_episode: int = 100,
        collision_reward: int | None = None,
        black_death: bool = True,
        layout: str | None = None
    ) -> None:
        super().__init__(num_agents=n_agents)
        assert (grid_shape[0] * grid_shape[1]) >= (n_agents + n_tokens * n_token_repeat)
        self.n_agents = n_agents
        self.n_tokens = n_tokens
        self.n_token_repeat = n_token_repeat
        self.grid_shape = grid_shape
        self.grid_shape_arr = jnp.array(self.grid_shape)
        self.fixed_map_seed = fixed_map_seed
        self.max_steps_in_episode = max_steps_in_episode
        self.collision_reward = collision_reward
        self.black_death = black_death
        self.n_buttons = 0

        self.agents = [f"agent_{i}" for i in range(self.n_agents)]

        self.init_state = None
        if layout is not None:
            self.init_state = self.parse(layout)
        self.num_agents = self.n_agents

        channel_dim = 1
        if self.init_state is not None: channel_dim += 2
        if self.n_tokens > 0: channel_dim += self.n_tokens
        if self.n_agents > 1: channel_dim += self.n_agents - 1
        if self.n_buttons > 0: channel_dim += 3 * self.n_buttons
        self.obs_shape = (channel_dim, *self.grid_shape)

        self.action_spaces = {
            agent: spaces.Discrete(len(ACTION_MAP))
            for agent in self.agents
        }
        self.observation_spaces = {
            agent: spaces.Box(low=0, high=1, shape=self.obs_shape, dtype=jnp.uint8)
            for agent in self.agents
        }

    @partial(jax.jit, static_argnums=(0,))
    def reset(
        self,
        key: chex.PRNGKey
    ) -> Tuple[Dict[str, chex.Array], TokenEnvState]:
        state = self.init_state
        if state is None: state = self.sample_init_state(key)
        obs = self.get_obs(state=state)
        return obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step_env(
        self,
        key: chex.PRNGKey,
        state: TokenEnvState,
        actions: Dict[str, chex.Array]
    ) -> Tuple[Dict[str, chex.Array], TokenEnvState, Dict[str, float], Dict[str, bool], Dict]:

        _actions = jnp.array([actions[agent] for agent in self.agents])

        # Move agents
        def move_agent(pos, a):
            return (pos + ACTION_MAP[a]) % self.grid_shape_arr
        new_agent_pos = jax.vmap(move_agent, in_axes=(0, 0))(state.agent_positions, _actions)
        new_agent_pos = jnp.where(state.is_alive[:, None], new_agent_pos, state.agent_positions)

        if self.init_state is not None:
            # Handle wall collisions
            def compute_wall_collisions(pos, wall_positions, is_wall_disabled):
                return jnp.any(
                    jnp.logical_and(
                        jnp.logical_not(is_wall_disabled),
                        jnp.all(
                            pos[None, :] == wall_positions # [N, 2]
                        , axis=-1), # [N,]
                    )
                , axis=-1) # [1,]
            wall_collisions = jax.vmap(compute_wall_collisions, in_axes=(0, None, None))(new_agent_pos, state.wall_positions, state.is_wall_disabled)
            new_agent_pos = jnp.where(wall_collisions[:, None], state.agent_positions, new_agent_pos)

        # Handle collisions
        # TODO: When collision_reward is not None, there might be unintended behavior.
        # +-------+-------+-------+-------+-------+-------+-------+
        # | 0     | #     | .     | #     | .     | #     | 2     |
        # +-------+-------+-------+-------+-------+-------+-------+
        # | .     | #     | .     | #     | 1     | #     | .     |
        # +-------+-------+-------+-------+-------+-------+-------+
        # | 4     | #     | 9     | 5     | 3     | #     | 7     |
        # +-------+-------+-------+-------+-------+-------+-------+
        # | .     | .     | 8     | #     | .     | #     | 7     |
        # +-------+-------+-------+-------+-------+-------+-------+
        # | 6     | #     | 2     | #     | 9     | #     | 3     |
        # +-------+-------+-------+-------+-------+-------+-------+
        # | 8     | #     | 0     | #     | .     | #     | A_1,5 |
        # +-------+-------+-------+-------+-------+-------+-------+
        # | .     | #     | 1     | #     | 4     | A_2   | A_0,6 |
        # +-------+-------+-------+-------+-------+-------+-------+
        # Action for agent_0
        # 3
        # Action for agent_1
        # 0
        # Action for agent_2
        # 1
        # Gives
        # {'agent_0': Array(-100., dtype=float32), 'agent_1': Array(-100., dtype=float32), 'agent_2': Array(-100., dtype=float32)}
        # {'__all__': Array(True, dtype=bool), 'agent_0': Array(True, dtype=bool), 'agent_1': Array(True, dtype=bool), 'agent_2': Array(True, dtype=bool)}
        def compute_collisions(mask):
            positions = jnp.where(mask[:, None], state.agent_positions, new_agent_pos)

            collision_grid = jnp.zeros(self.grid_shape)
            collision_grid, _ = jax.lax.scan(
                lambda grid, pos: (grid.at[pos[0], pos[1]].add(1), None),
                collision_grid,
                positions,
            )

            collision_mask = collision_grid > 1

            collisions = jax.vmap(lambda p: collision_mask[p[0], p[1]])(positions)
            return jnp.logical_and(state.is_alive, collisions)

        collisions = jax.lax.while_loop(
            lambda mask: jnp.any(compute_collisions(mask)),
            lambda mask: jnp.logical_or(mask, compute_collisions(mask)),
            jnp.zeros((self.n_agents,), dtype=bool)
        )

        if self.collision_reward is None:
            new_agent_pos = jnp.where(collisions[:, None], state.agent_positions, new_agent_pos)
            collisions = jnp.full(collisions.shape, False)

        # Handle swaps
        def compute_swaps(original_positions, new_positions):
            original_pos_expanded = jnp.expand_dims(original_positions, axis=0)
            new_pos_expanded = jnp.expand_dims(new_positions, axis=1)

            swap_mask = (original_pos_expanded == new_pos_expanded).all(axis=-1)
            swap_mask = jnp.fill_diagonal(swap_mask, False, inplace=False)

            swap_pairs = jnp.logical_and(swap_mask, swap_mask.T)

            swaps = jnp.any(swap_pairs, axis=0)
            return swaps

        swaps = compute_swaps(state.agent_positions, new_agent_pos)
        new_agent_pos = jnp.where(swaps[:, None], state.agent_positions, new_agent_pos)

        _rewards = jnp.zeros((self.n_agents,), dtype=jnp.float32)
        if self.collision_reward is not None:
            _rewards = jnp.where(jnp.logical_and(state.is_alive, collisions), self.collision_reward, _rewards)
        rewards = {agent: _rewards[i] for i, agent in enumerate(self.agents)}

        is_wall_disabled = jnp.empty((0, 2), dtype=bool)
        if self.init_state is not None:
            is_wall_disabled = self.compute_disabled_walls(new_agent_pos, state.wall_positions, state.button_positions)

        new_state = TokenEnvState(
            agent_positions=new_agent_pos,
            token_positions=state.token_positions,
            wall_positions=state.wall_positions,
            is_wall_disabled=is_wall_disabled,
            button_positions=state.button_positions,
            is_alive=jnp.logical_and(state.is_alive, jnp.logical_not(collisions)),
            time=state.time + 1
        )

        _dones = jnp.logical_or(collisions, new_state.time >= self.max_steps_in_episode)
        dones = {a: _dones[i] for i, a in enumerate(self.agents)}
        dones.update({"__all__": jnp.all(_dones)})

        obs = self.get_obs(state=new_state)
        info = {}

        return obs, new_state, rewards, dones, info

    @partial(jax.jit, static_argnums=(0,))
    def get_obs(
        self,
        state: TokenEnvState
    ) -> Dict[str, chex.Array]:

        def obs_for_agent(i):
            base = jnp.zeros(self.obs_shape, dtype=jnp.uint8)
            # ref = self.grid_shape_arr // 2
            ref = jnp.array([0, 0])
            offset = ref - state.agent_positions[i]
            idx_offset = 0
            b = base

            def place_agent(val):
                rel = (state.agent_positions[i] + offset) % self.grid_shape_arr
                return val.at[idx_offset, rel[0], rel[1]].set(1) # Is agent?
            b = place_agent(b)
            idx_offset += 1

            if self.init_state is not None:
                def place_wall(val):
                    rel = (state.wall_positions + offset) % self.grid_shape_arr
                    return val.at[
                        idx_offset, rel[:, 0], rel[:, 1]
                    ].set(1).at[ # Is wall?
                        idx_offset + 1, rel[:, 0], rel[:, 1]
                    ].set(jnp.logical_not(state.is_wall_disabled).astype(jnp.uint8)) # Is wall blocking?
                b = place_wall(b)
                idx_offset += 2

            if self.n_tokens > 0:
                def place_token(token_idx, val):
                    rel = (state.token_positions[token_idx] + offset) % self.grid_shape_arr
                    return val.at[idx_offset + token_idx, rel[:, 0], rel[:, 1]].set(1) # Is token?
                b = jax.lax.fori_loop(0, self.n_tokens, place_token, b)
                idx_offset += self.n_tokens

            if self.n_agents > 1:
                def place_other(other_idx, val):
                    rel = (state.agent_positions[other_idx + (other_idx >= i)] + offset) % self.grid_shape_arr
                    return val.at[idx_offset + other_idx, rel[0], rel[1]].set(1) # Is other agent?
                b = jax.lax.fori_loop(0, self.n_agents - 1, place_other, b)
                idx_offset += self.n_agents - 1

            if self.n_buttons > 0:
                def place_button(button_idx, val):
                    is_door = jnp.any(
                        jnp.all(
                            state.button_positions[button_idx][:, None, :] == state.wall_positions[None, :, :]
                        , axis=-1)
                    , axis=-1) # Buttons are considered doors if they are in a wall.
                    rel = (state.button_positions[button_idx] + offset) % self.grid_shape_arr
                    return val.at[
                        idx_offset + 3 * button_idx, rel[:, 0], rel[:, 1]
                    ].set(1).at[ # Is button-door pair?
                        idx_offset + 3 * button_idx + 1, rel[:, 0], rel[:, 1]
                    ].set(jnp.logical_not(is_door).astype(jnp.uint8)).at[ # Is button?
                        idx_offset + 3 * button_idx + 2, rel[:, 0], rel[:, 1]
                    ].set(is_door.astype(jnp.uint8)) # Is door?

                b = jax.lax.fori_loop(0, self.n_buttons, place_button, b)

            return jnp.where(jnp.logical_or(jnp.logical_not(self.black_death), state.is_alive[i]), b, base)

        obs = jax.vmap(obs_for_agent)(jnp.arange(self.n_agents))
        return {agent: obs[i] for i, agent in enumerate(self.agents)}

    @partial(jax.jit, static_argnums=(0,))
    def label_f(self, state: TokenEnvState) -> Dict[str, int]:

        diffs = state.agent_positions[:, None, None, :] - state.token_positions[None, :, :, :]
        matches = jnp.all(diffs == 0, axis=-1)
        matches_any = jnp.any(matches, axis=-1)

        has_match = jnp.any(matches_any, axis=1)
        token_idx = jnp.argmax(matches_any, axis=1)

        agent_token_matches = jnp.where(jnp.logical_and(has_match, state.is_alive), token_idx, -1)

        return {self.agents[agent_idx]: token_idx for agent_idx, token_idx in enumerate(agent_token_matches)}

    @partial(jax.jit, static_argnums=(0,))
    def sample_init_state(
        self,
        key: chex.PRNGKey
    ) -> Tuple[Dict[str, chex.Array], TokenEnvState]:
        if self.fixed_map_seed is not None:
            key = jax.random.PRNGKey(self.fixed_map_seed)

        grid_points = jnp.stack(jnp.meshgrid(jnp.arange(self.grid_shape[0]), jnp.arange(self.grid_shape[1])), -1)
        grid_flat = grid_points.reshape(-1, 2)

        key, subkey = jax.random.split(key)
        perm = jax.random.permutation(subkey, grid_flat.shape[0])

        agent_positions = grid_flat[perm][:self.n_agents]
        token_positions = grid_flat[perm][self.n_agents: self.n_agents + self.n_tokens * self.n_token_repeat].reshape(self.n_tokens, self.n_token_repeat, 2)

        return TokenEnvState(
            agent_positions=agent_positions,
            token_positions=token_positions,
            wall_positions=jnp.empty((0, 2), dtype=jnp.int32),
            is_wall_disabled=jnp.empty((0, 2), dtype=bool),
            button_positions=jnp.empty((0, 2), dtype=jnp.int32),
            is_alive=jnp.ones((self.n_agents,), dtype=bool),
            time=0
        )

    @partial(jax.jit, static_argnums=(0,))
    def compute_disabled_walls(self, agent_positions, wall_positions, button_positions):
        def _compute_on_buttons(button_pos, agent_positions):
            return jnp.any(
                jnp.any(
                    jnp.all(
                        button_pos[:, None, :] == agent_positions[None, :, :] # [M, N, 2]
                    , axis=-1) # [M, N,]
                , axis=-1) # [M,]
            , axis=-1) # [1,]
        on_buttons = jax.vmap(_compute_on_buttons, in_axes=(0, None))(button_positions, agent_positions)
        def _compute_disabled_walls(wall_pos, on_buttons, button_positions):
            # Compare each wall_pos to each button coordinate
            eq = jnp.all(button_positions == wall_pos, axis=-1)  # (n_buttons, n_button_repeat)
            # A wall is disabled if *any* matching button is pressed
            return jnp.any(jnp.logical_and(on_buttons, jnp.any(eq, axis=-1)))
        return jax.vmap(_compute_disabled_walls, in_axes=(0, None, None))(wall_positions, on_buttons, button_positions)

    def parse(self, layout: str) -> TokenEnvState:
        # Example layout:
        # [ 8 ][   ][   ][   ][   ][   ][   ][ # ][ 0 ][   ][   ][ 1 ]
        # [   ][   ][   ][   ][   ][   ][   ][ # ][   ][   ][   ][   ]
        # [   ][   ][ b ][   ][   ][   ][   ][ # ][   ][   ][   ][   ]
        # [   ][   ][   ][   ][   ][   ][   ][ # ][ 3 ][   ][   ][ 2 ]
        # [   ][   ][   ][   ][   ][   ][   ][ # ][ # ][ # ][#,a][ # ]
        # [ A ][   ][   ][   ][   ][   ][   ][   ][   ][   ][   ][   ]
        # [ B ][   ][   ][   ][   ][   ][   ][   ][   ][   ][   ][   ]
        # [   ][   ][   ][   ][   ][   ][   ][ # ][ # ][ # ][#,b][ # ]
        # [   ][   ][   ][   ][   ][   ][   ][ # ][ 4 ][   ][   ][ 5 ]
        # [   ][   ][ a ][   ][   ][   ][   ][ # ][   ][   ][   ][   ]
        # [   ][   ][   ][   ][   ][   ][   ][ # ][   ][   ][   ][   ]
        # [ 9 ][   ][   ][   ][   ][   ][   ][ # ][ 7 ][   ][   ][ 6 ]
        # where each [] indicates a cell, uppercase letters indicate agents,
        # e.g., A and B, and lower case letters indicate "sync" points which
        # are more like doors and buttons that open the doors, eg [#|a] is a
        # door that is open if there is an agent on the cell [ a ] and closed;
        # otherwise, and cells with # indicate walls.

        # --- parse into a 2D list of cell contents (strings inside brackets) ---
        lines = [ln.strip() for ln in layout.strip().splitlines() if ln.strip()]
        rows: list[list[str]] = []

        for ln in lines:
            cells = []
            i = 0
            while i < len(ln):
                if ln[i] == "[":
                    j = ln.find("]", i + 1)
                    if j == -1:
                        raise ValueError("Malformed layout: missing closing ']' in a row.")
                    cells.append(ln[i + 1:j].strip())
                    i = j + 1
                else:
                    i += 1
            if cells:
                rows.append(cells)

        if not rows:
            raise ValueError("Parsed layout is empty.")

        H = len(rows)
        W = len(rows[0])
        if any(len(r) != W for r in rows):
            raise ValueError("All rows in the layout must have the same number of cells.")

        # --- collect info ---
        wall_positions: list[tuple[int, int]] = []
        agent_positions: dict[int, tuple[int, int]] = {}
        token_positions: dict[int, list[tuple[int, int]]] = {}
        button_positions: dict[int, list[tuple[int, int]]] = {}
        max_token_id = -1
        max_button_id = -1

        def _is_wall(c: str) -> bool:
            return "#" == c

        def _is_agent(c: str) -> bool:
            is_agent_in_cell = [
                a == c
                for a in [chr(ord("A") + i) for i in range(ord("Z") - ord("A") + 1)]
            ]
            return sum(is_agent_in_cell) > 0

        def _get_agent_idx(c: str) -> bool:
            is_agent_in_cell = [
                a == c
                for a in [chr(ord("A") + i) for i in range(ord("Z") - ord("A") + 1)]
            ]
            return np.argmax(is_agent_in_cell)

        def _is_token(c: str) -> bool:
            try:
                int(c)
                return True
            except ValueError:
                return False

        def _is_button(c: str) -> bool:
            is_button_in_cell = [
                a == c
                for a in [chr(ord("a") + i) for i in range(ord("z") - ord("a") + 1)]
            ]
            return sum(is_button_in_cell) > 0

        def _get_button_idx(c: str) -> bool:
            is_button_in_cell = [
                a == c
                for a in [chr(ord("a") + i) for i in range(ord("z") - ord("a") + 1)]
            ]
            return np.argmax(is_button_in_cell)

        for r in range(H):
            for c in range(W):
                cell = rows[r][c]
                has_wall = False
                has_agent = False
                has_token = False
                has_button = False
                for content in cell.split(","):

                    if _is_wall(content):
                        if has_wall:
                            raise ValueError(f"One wall per cell.")
                        has_wall = True
                        wall_positions.append((r, c))

                    if _is_agent(content):
                        if has_agent:
                            raise ValueError(f"One agent per cell.")
                        has_agent = True
                        idx = _get_agent_idx(content)
                        if idx in agent_positions:
                            raise ValueError(f"Duplicate placement for agent '{content}'.")
                        agent_positions[idx] = (r, c)

                    if _is_token(content):
                        if has_token:
                            raise ValueError(f"One token per cell.")
                        has_token = True
                        tok_id = int(content)
                        max_token_id = max(max_token_id, tok_id)
                        token_positions.setdefault(tok_id, []).append((r, c))

                    if _is_button(content):
                        if has_button:
                            raise ValueError(f"One button per cell.")
                        has_button = True
                        idx = _get_button_idx(content)
                        max_button_id = max(max_button_id, idx)
                        button_positions.setdefault(idx, []).append((r, c))

                assert not (has_wall and has_agent)
                assert not (has_wall and has_token)
                assert not (has_token and has_button)

        # --- override environment settings ---
        self.grid_shape = (H, W)
        self.grid_shape_arr = jnp.array(self.grid_shape)

        self.n_agents = len(agent_positions)
        self.agents = [f"agent_{i}" for i in range(self.n_agents)]

        self.n_tokens = max_token_id + 1 if max_token_id >= 0 else 0
        self.n_token_repeat = max((len(v) for v in token_positions.values()), default=0)
        token_positions_np = np.full((self.n_tokens, self.n_token_repeat, 2), -1, dtype=np.int32)
        for tid in range(self.n_tokens):
            coords = token_positions.get(tid, [])
            for k, (r, c) in enumerate(coords[: self.n_token_repeat]):
                token_positions_np[tid, k] = (r, c)

        agent_positions_np = np.full((self.n_agents, 2), -1, dtype=np.int32)
        for idx, pos in agent_positions.items():
            agent_positions_np[idx] = pos

        wall_positions_np = np.array(wall_positions, dtype=np.int32) if wall_positions else np.empty((0, 2), dtype=np.int32)

        self.n_buttons = max_button_id + 1 if max_button_id >= 0 else 0
        self.n_button_repeat = max((len(v) for v in button_positions.values()), default=0)
        button_positions_np = np.full((self.n_buttons, self.n_button_repeat, 2), -1, dtype=np.int32)
        for bid in range(self.n_buttons):
            coords = button_positions.get(bid, [])
            for k, (r, c) in enumerate(coords[: self.n_button_repeat]):
                button_positions_np[bid, k] = (r, c)

        agent_positions_jnp = jnp.array(agent_positions_np)
        wall_positions_jnp = jnp.array(wall_positions_np)
        button_positions_jnp = jnp.array(button_positions_np)

        # --- return state ---
        return TokenEnvState(
            agent_positions=agent_positions_jnp,
            token_positions=jnp.array(token_positions_np),
            wall_positions=wall_positions_jnp,
            is_wall_disabled=self.compute_disabled_walls(agent_positions_jnp, wall_positions_jnp, button_positions_jnp),
            button_positions=button_positions_jnp,
            is_alive=jnp.ones((self.n_agents,), dtype=bool),
            time=0,
        )

    def render(self, state: TokenEnvState):
        empty_cell = "."
        wall_cell = "#"
        grid = np.full(self.grid_shape, empty_cell, dtype=object)

        for disabled, pos in zip(state.is_wall_disabled, state.wall_positions):
            if not disabled:
                grid[pos[0], pos[1]] = f"{wall_cell}"

        for token, positions in enumerate(state.token_positions):
            for pos in positions:
                grid[pos[0], pos[1]] = f"{token}"

        for button, positions in enumerate(state.button_positions):
            for pos in positions:
                current = grid[pos[0], pos[1]]
                if current == empty_cell:
                    grid[pos[0], pos[1]] = f"b_{button}"
                else:
                    grid[pos[0], pos[1]] = f"{current},b_{button}"

        for agent in range(self.n_agents):
            pos = state.agent_positions[agent]
            current = grid[pos[0], pos[1]]
            if current == empty_cell:
                grid[pos[0], pos[1]] = f"A_{agent}"
            else:
                grid[pos[0], pos[1]] = f"A_{agent},{current}"

        max_width = max(len(str(cell)) for row in grid for cell in row)

        out = ""
        h_line = "+" + "+".join(["-" * (max_width + 2) for _ in range(self.grid_shape[1])]) + "+"
        out += h_line + "\n"
        for row in grid:
            row_str = "| " + " | ".join(f"{str(cell):<{max_width}}" for cell in row) + " |"
            out += row_str + "\n"
            out += h_line + "\n"

        print(out)

