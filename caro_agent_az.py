"""
caro_agent_az.py — AlphaZero Agent (Multi-GPU + Replay Buffer)

Tính năng:
  1. Replay Buffer — gom nhiều ván, shuffle, train batch lớn
  2. Multi-GPU MirroredStrategy
  3. FCN 3-channel, parallel MCTS, batch training
  4. Pretrain from greedy, copy_weights_from, save/load
"""

import tensorflow as tf
import numpy as np
from IPython import display, get_ipython
from IPython.display import Image
from io import BytesIO
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from config import (BOARD_SIZE, WIN_LEN, LR, WEIGHT_DECAY,
                    C_PUCT, N_SIM, N_PARALLEL, DIR_ALPHA, DIR_WEIGHT,
                    BUFFER_SIZE, TRAIN_BATCH, TRAIN_EPOCHS)
from model_fcn import build_model, reshape_policy, ResBlock, transfer_weights


# ================================================================
# ENCODE STATE
# ================================================================

def encode_state(board, player):
    my    = (board == player).astype(np.float32)
    enemy = (board == -player).astype(np.float32)
    empty = (board == 0).astype(np.float32)
    return np.stack([my, enemy, empty], axis=-1)


# ================================================================
# MCTS NODE
# ================================================================

class MCTSNode:
    __slots__ = ["N", "W", "P", "children"]
    def __init__(self, prior):
        self.N = 0; self.W = 0.0; self.P = prior; self.children = {}

    @property
    def Q(self):
        return self.W / self.N if self.N > 0 else 0.0
    def is_leaf(self):
        return not self.children
    def ucb(self, n_parent, c_puct):
        return self.Q + c_puct * self.P * (n_parent ** 0.5) / (1.0 + self.N)


# ================================================================
# MCTS — Parallel + virtual loss
# ================================================================

class MCTS:
    def __init__(self, model, board_size=BOARD_SIZE, n_sim=N_SIM,
                 n_parallel=N_PARALLEL, c_puct=C_PUCT,
                 dir_alpha=DIR_ALPHA, dir_weight=DIR_WEIGHT):
        self.model      = model
        self.board_size = board_size
        self.n_sim      = n_sim
        self.n_parallel = n_parallel
        self.c_puct     = c_puct
        self.dir_alpha  = dir_alpha
        self.dir_weight = dir_weight

    def get_pi(self, board, player, temperature=1.0):
        root = MCTSNode(prior=1.0)
        self._expand_single(root, board, player)
        self._add_dirichlet(root)

        sims_done = 0
        while sims_done < self.n_sim:
            batch = min(self.n_parallel, self.n_sim - sims_done)
            self._simulate_batch(root, board, player, batch)
            sims_done += batch

        bs     = self.board_size
        visits = np.zeros((bs, bs), dtype=np.float32)
        for action, child in root.children.items():
            r, c = divmod(action, bs)
            visits[r, c] = child.N

        if temperature == 0:
            pi = np.zeros((bs, bs), dtype=np.float32)
            pi[np.unravel_index(np.argmax(visits), (bs, bs))] = 1.0
        else:
            v     = visits ** (1.0 / temperature)
            total = v.sum()
            pi    = v / total if total > 0 else visits
        return pi

    def _simulate_batch(self, root, board, player, batch_size):
        pending = []
        for _ in range(batch_size):
            node = root; path = []; b = board.copy(); p = player; terminal = None
            while not node.is_leaf():
                best_a = max(node.children,
                    key=lambda a: node.children[a].ucb(node.N, self.c_puct))
                path.append((node, best_a))
                child = node.children[best_a]
                child.N += 1; child.W -= 1.0
                node = child
                r, c = divmod(best_a, self.board_size)
                b[r, c] = p; p *= -1
                if self._check_win(b, r, c, -p): terminal = -1.0; break
                if np.all(b != 0): terminal = 0.0; break
            pending.append((path, node, b, p, terminal))

        to_expand = []
        for i, (path, node, b, p, terminal) in enumerate(pending):
            if terminal is not None:
                self._undo_vl(path); self._backup(path, node, terminal)
            else:
                to_expand.append(i)
        if not to_expand: return

        states, valids = [], []
        for i in to_expand:
            _, node, b, p, _ = pending[i]
            valid = self._get_valid_near(b); valids.append(valid)
            states.append(encode_state(b, p) if valid.any()
                else np.zeros((self.board_size, self.board_size, 3), dtype=np.float32))

        p_b, v_b = self.model(np.array(states, dtype=np.float32), training=False)
        p_b = p_b.numpy(); v_b = v_b.numpy()

        for j, i in enumerate(to_expand):
            path, node, b, p, _ = pending[i]
            valid = valids[j]; self._undo_vl(path)
            if not valid.any(): self._backup(path, node, 0.0); continue
            p_2d = reshape_policy(p_b[j:j+1], self.board_size)[0].astype(np.float64)
            v = float(v_b[j, 0])
            p_2d *= valid.astype(np.float64)
            s = p_2d.sum()
            p_2d = p_2d / s if s > 0 else valid.astype(np.float64) / valid.sum()
            bs = self.board_size
            for r in range(bs):
                for c in range(bs):
                    if valid[r, c]:
                        node.children[r*bs+c] = MCTSNode(prior=float(p_2d[r, c]))
            self._backup(path, node, v)

    def _undo_vl(self, path):
        for parent, action in path:
            child = parent.children[action]; child.N -= 1; child.W += 1.0

    def _expand_single(self, node, board, player):
        valid = self._get_valid_near(board)
        if not valid.any(): return 0.0
        state = encode_state(board, player)[np.newaxis]
        pf, vo = self.model(state, training=False)
        p = reshape_policy(pf.numpy(), self.board_size)[0].astype(np.float64)
        v = float(vo[0, 0])
        p *= valid.astype(np.float64); s = p.sum()
        p = p / s if s > 0 else valid.astype(np.float64) / valid.sum()
        bs = self.board_size
        for r in range(bs):
            for c in range(bs):
                if valid[r, c]:
                    node.children[r*bs+c] = MCTSNode(prior=float(p[r, c]))
        return v

    def _get_valid_near(self, board, radius=3):
        bs = self.board_size; has = (board != 0)
        if not has.any():
            mask = np.zeros((bs, bs), dtype=bool)
            c = bs // 2; r = radius * 2
            mask[max(0,c-r):c+r+1, max(0,c-r):c+r+1] = True
        else:
            mask = np.zeros((bs, bs), dtype=bool)
            rs, cs = np.where(has)
            for sr, sc in zip(rs, cs):
                mask[max(0,sr-radius):min(bs,sr+radius+1),
                     max(0,sc-radius):min(bs,sc+radius+1)] = True
        return (board == 0) & mask

    def _backup(self, path, leaf, v):
        if not path: leaf.N += 1; leaf.W += v; return
        for parent, action in reversed(path):
            child = parent.children[action]; child.N += 1; child.W += v; v = -v
        path[0][0].N += 1

    def _add_dirichlet(self, root):
        actions = list(root.children.keys())
        if not actions: return
        noise = np.random.dirichlet([self.dir_alpha] * len(actions))
        for a, n in zip(actions, noise):
            c = root.children[a]
            c.P = (1.0 - self.dir_weight) * c.P + self.dir_weight * n

    def _check_win(self, board, row, col, player):
        for dr, dc in [(0,1),(1,0),(1,1),(1,-1)]:
            count = 1
            for sign in [1, -1]:
                r, c = row+sign*dr, col+sign*dc
                while 0 <= r < self.board_size and 0 <= c < self.board_size \
                      and board[r, c] == player:
                    count += 1; r += sign*dr; c += sign*dc
            if count >= WIN_LEN: return True
        return False


# ================================================================
# REPLAY BUFFER
# ================================================================

class ReplayBuffer:
    """
    Ring buffer lưu (state, pi, z).
    Khi đầy, vị trí cũ nhất bị ghi đè.
    """
    def __init__(self, max_size=BUFFER_SIZE):
        self.max_size = max_size
        self.states   = []     # (bs, bs, 3) float32
        self.pis      = []     # (bs*bs,)    float32
        self.zs       = []     # float32
        self._pos     = 0      # vị trí ghi tiếp theo

    def __len__(self):
        return len(self.states)

    def add(self, state, pi, z):
        """Thêm 1 position vào buffer."""
        entry = (state, pi.flatten().astype(np.float32), np.float32(z))
        if len(self.states) < self.max_size:
            self.states.append(entry[0])
            self.pis.append(entry[1])
            self.zs.append(entry[2])
        else:
            self.states[self._pos] = entry[0]
            self.pis[self._pos]    = entry[1]
            self.zs[self._pos]     = entry[2]
        self._pos = (self._pos + 1) % self.max_size

    def add_game(self, steps, result):
        """
        Thêm tất cả positions từ 1 ván.
        steps: [(state, pi), ...]
        result: "win"/"lose"/"draw"/"invalid"
        """
        z_map = {"win": 1.0, "lose": -1.0, "draw": 0.0, "invalid": -1.0}
        z     = z_map.get(result, 0.0)
        for state, pi in steps:
            self.add(state, pi, z)

    def sample_batches(self, batch_size, epochs=1):
        """
        Shuffle toàn bộ buffer, chia thành batches.
        Batch cuối bị bỏ nếu nhỏ hơn batch_size (drop_last)
        → đảm bảo mọi batch cùng size, chia đều cho multi-GPU.
        """
        n       = len(self.states)
        indices = np.arange(n)

        for _ in range(epochs):
            np.random.shuffle(indices)
            for start in range(0, n - batch_size + 1, batch_size):
                idxs = indices[start:start + batch_size]

                s_batch = np.array([self.states[i] for i in idxs], dtype=np.float32)
                p_batch = np.array([self.pis[i]    for i in idxs], dtype=np.float32)
                z_batch = np.array([self.zs[i]     for i in idxs], dtype=np.float32)

                yield s_batch, p_batch, z_batch

    def clear(self):
        self.states.clear(); self.pis.clear(); self.zs.clear()
        self._pos = 0


# ================================================================
# AGENT — Multi-GPU + Replay Buffer
# ================================================================

class CaroAgent:
    """
    AlphaZero agent.

    Interface:
        get_action(board, player)          → (row, col)
        train_on_game(result)              — lưu data vào replay buffer
        train_buffer(batch_size, epochs)   — train từ buffer
        pretrain_from_greedy(greedy, ...)
        copy_weights_from(other_agent)
        save(path) / load(path)
    """

    def __init__(self, board_size=BOARD_SIZE, base_filters=128,
                 n_res=10, n_sim=N_SIM, n_parallel=N_PARALLEL,
                 buffer_size=BUFFER_SIZE, strategy=None):
        self.board_size   = board_size
        self.base_filters = base_filters
        self.n_res        = n_res

        # Strategy: dùng chung nếu truyền vào, tự tạo nếu không
        if strategy is not None:
            self.strategy = strategy
        else:
            self.strategy = tf.distribute.MirroredStrategy()
        n_gpu = self.strategy.num_replicas_in_sync
        print(f"Strategy: {n_gpu} replica(s)")

        # Model + optimizer trong scope
        with self.strategy.scope():
            self.model     = build_model(board_size, base_filters, n_res)
            self.optimizer = tf.keras.optimizers.Adam(LR, weight_decay=WEIGHT_DECAY)
            _d = np.zeros((1, board_size, board_size, 3), dtype=np.float32)
            self.model(_d, training=False)
            self.model(_d, training=True)
            self.optimizer.build(self.model.trainable_variables)

        self.mcts   = MCTS(self.model, board_size, n_sim, n_parallel)
        self.buffer = ReplayBuffer(buffer_size)

        self._steps       = []     # steps ván hiện tại
        self.game_count   = 0
        self.game_history = []

    # ── GET ACTION ───────────────────────────────────────────────

    def get_action(self, board, player, temperature=None):
        if temperature is None:
            temperature = 1.0 if len(self._steps) < 30 else 0.0

        pi    = self.mcts.get_pi(board, player, temperature)
        valid = (board == 0)
        pi    = pi * valid
        s     = pi.sum()
        pi    = pi / s if s > 0 else valid.astype(float) / valid.sum()

        self._steps.append((encode_state(board, player), pi.copy()))

        if temperature == 0:
            idx = int(np.argmax(pi))
        else:
            idx = np.random.choice(len(pi.flatten()), p=pi.flatten())
        return divmod(idx, self.board_size)

    # ── COMPILED TRAIN STEP ──────────────────────────────────────

    @tf.function
    def _train_step(self, states, pis, zs):
        with tf.GradientTape() as tape:
            p_flat, v = self.model(states, training=True)
            policy_loss = -tf.reduce_sum(pis * tf.math.log(p_flat + 1e-8), axis=1)
            value_loss  = tf.square(v[:, 0] - zs)
            per_example = policy_loss + value_loss
            loss = tf.nn.compute_average_loss(per_example)
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss

    @tf.function
    def _pretrain_step(self, states, pis):
        with tf.GradientTape() as tape:
            p_flat, _ = self.model(states, training=True)
            per_example = -tf.reduce_sum(pis * tf.math.log(p_flat + 1e-8), axis=1)
            loss = tf.nn.compute_average_loss(per_example)
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss

    # ── TRAIN ON GAME — LƯU VÀO BUFFER ──────────────────────────

    def train_on_game(self, result):
        """
        Lưu data ván vừa chơi vào replay buffer.
        KHÔNG train ngay — gọi train_buffer() khi muốn train.
        """
        if not self._steps:
            return

        # Lưu vào game_history để tracking
        self.game_count += 1
        self.game_history.append({
            "game"        : self.game_count,
            "result"      : result,
            "total_steps" : len(self._steps),
            "total_loss"  : None,
            "avg_pi_max"  : round(float(np.mean([pi.max() for _, pi in self._steps])), 6),
        })

        # Lưu vào replay buffer
        self.buffer.add_game(self._steps, result)
        self._steps = []

    # ── TRAIN FROM BUFFER ────────────────────────────────────────

    def train_buffer(self, batch_size=TRAIN_BATCH, epochs=TRAIN_EPOCHS):
        """
        Shuffle toàn bộ replay buffer → chia batch → train.
        Batch_size tự động giảm nếu buffer quá nhỏ.
        """
        buf_len = len(self.buffer)
        if buf_len == 0:
            return 0.0

        # Nếu buffer nhỏ hơn batch_size → dùng buffer size (tránh skip hết)
        actual_bs = min(batch_size, buf_len)

        total_loss = 0.0
        n_batches  = 0

        for s_batch, p_batch, z_batch in self.buffer.sample_batches(actual_bs, epochs):
            s_t = tf.constant(s_batch)
            p_t = tf.constant(p_batch)
            z_t = tf.constant(z_batch)

            per_replica = self.strategy.run(
                self._train_step, args=(s_t, p_t, z_t)
            )
            loss = self.strategy.reduce(
                tf.distribute.ReduceOp.SUM, per_replica, axis=None
            )
            total_loss += float(loss.numpy())
            n_batches  += 1

        avg_loss = total_loss / max(n_batches, 1)

        # Ghi loss vào game cuối
        if self.game_history:
            self.game_history[-1]["total_loss"] = round(avg_loss, 6)

        return avg_loss

    # ── COPY WEIGHTS ─────────────────────────────────────────────

    def copy_weights_from(self, other_agent):
        transfer_weights(other_agent.model, self.model)
        with self.strategy.scope():
            self.optimizer = tf.keras.optimizers.Adam(LR, weight_decay=WEIGHT_DECAY)
            self.optimizer.build(self.model.trainable_variables)
        self.mcts.model = self.model
        print(f"✓ Copy weights: {other_agent.board_size}×{other_agent.board_size}"
              f" → {self.board_size}×{self.board_size}")

    # ── PRETRAIN FROM GREEDY ─────────────────────────────────────

    def pretrain_from_greedy(self, greedy, n_games=100,
                             batch_size=64, temperature=2.0,
                             random_opening=4, skip_first=4):
        print(f"Thu thập positions từ {n_games} ván greedy vs greedy...")
        all_pos = self._collect_greedy_games(greedy, n_games, random_opening, skip_first)
        n_pos = len(all_pos)
        print(f"  → {n_pos} positions")
        if n_pos == 0: return 0.0

        print(f"Train {n_pos} positions, batch={batch_size}")
        np.random.shuffle(all_pos)
        total_loss_sum = 0.0

        for start in range(0, n_pos, batch_size):
            end = min(start + batch_size, n_pos)
            batch = all_pos[start:end]
            sb, pb = [], []
            for board, player in batch:
                scores = self._greedy_scores(greedy, board, player)
                pi = self._scores_to_pi(scores, temperature)
                sb.append(encode_state(board, player))
                pb.append(pi.flatten())

            st = tf.constant(np.array(sb), dtype=tf.float32)
            pt = tf.constant(np.array(pb), dtype=tf.float32)
            per_rep = self.strategy.run(self._pretrain_step, args=(st, pt))
            loss = self.strategy.reduce(tf.distribute.ReduceOp.SUM, per_rep, axis=None)
            total_loss_sum += float(loss.numpy()) * len(batch)

            if (start // batch_size + 1) % 20 == 0 or end >= n_pos:
                print(f"  [{end}/{n_pos}]  avg loss = {total_loss_sum / end:.4f}")

        avg = total_loss_sum / n_pos
        print(f"✓ Pretrain xong — {n_games} ván → {n_pos} positions — avg loss = {avg:.4f}")
        return avg

    def _collect_greedy_games(self, greedy, n_games, random_opening, skip_first):
        bs = self.board_size; all_pos = []
        for _ in range(n_games):
            board = np.zeros((bs, bs), dtype=np.float32)
            turn = 1; log = []; done = False
            for _ in range(random_opening):
                empties = np.argwhere(board == 0)
                if len(empties) == 0: break
                r, c = empties[np.random.randint(len(empties))]
                board[r, c] = turn; log.append((turn, int(r), int(c)))
                if self._check_win_simple(board, turn, int(r), int(c)):
                    done = True; break
                turn *= -1
            if done: continue
            while not done:
                r, c = greedy.get_action(board, turn)
                if not (0 <= r < bs and 0 <= c < bs) or board[r, c] != 0: break
                board[r, c] = turn; log.append((turn, r, c))
                if self._check_win_simple(board, turn, r, c): done = True; break
                if np.all(board != 0): done = True; break
                turn *= -1
            if len(log) <= skip_first + 1: continue
            rb = np.zeros((bs, bs), dtype=np.float32)
            for si, (pl, mr, mc) in enumerate(log):
                if si >= skip_first and si < len(log) - 1:
                    all_pos.append((rb.copy(), pl))
                rb[mr, mc] = pl
        return all_pos

    def _check_win_simple(self, board, player, row, col):
        for dr, dc in [(0,1),(1,0),(1,1),(1,-1)]:
            count = 1
            for sign in [1, -1]:
                r, c = row+sign*dr, col+sign*dc
                while 0 <= r < self.board_size and 0 <= c < self.board_size \
                      and board[r, c] == player:
                    count += 1; r += sign*dr; c += sign*dc
            if count >= WIN_LEN: return True
        return False

    def _greedy_scores(self, greedy, board, player):
        bs = self.board_size; state = board * player
        scores = np.zeros((bs, bs), dtype=np.float64)
        for r in range(bs):
            for c in range(bs):
                if state[r, c] != 0: continue
                score, instant = greedy._score_cell(state, r, c)
                if instant: score = 1e6
                scores[r, c] = score
        return scores

    def _scores_to_pi(self, scores, temperature):
        bs = self.board_size; flat = scores.flatten()
        mask = flat > 0
        if not mask.any():
            pi = np.zeros((bs, bs), dtype=np.float64)
            pi[scores == 0] = 1.0
            s = pi.sum(); return pi / s if s > 0 else pi
        logits = np.full_like(flat, -1e9)
        logits[mask] = flat[mask] / temperature
        logits -= logits.max()
        exp = np.exp(logits); exp[~mask] = 0.0
        s = exp.sum()
        return (exp / s if s > 0 else exp).reshape(bs, bs)

    # ── PLOT ─────────────────────────────────────────────────────

    def plot_training(self, smooth_window=20):
        if len(self.game_history) < 2: return
        display.clear_output(wait=True)
        games   = [g["game"]       for g in self.game_history]
        losses  = [g["total_loss"] for g in self.game_history if g["total_loss"] is not None]
        l_games = [g["game"]       for g in self.game_history if g["total_loss"] is not None]
        pi_maxs = [g["avg_pi_max"] for g in self.game_history]
        results = [g["result"]     for g in self.game_history]
        n_steps = [g["total_steps"] for g in self.game_history]
        wr = []
        for i in range(len(games)):
            ch = results[max(0, i-smooth_window+1):i+1]
            wr.append(sum(1 for r in ch if r=="win") / len(ch))

        # ── Text log (luôn hiện trong Kaggle logs) ──
        last_loss = losses[-1] if losses else 0
        last_wr   = wr[-1] if wr else 0
        last_pi   = pi_maxs[-1] if pi_maxs else 0
        wins  = sum(1 for r in results if r == "win")
        loses = sum(1 for r in results if r == "lose")
        invs  = sum(1 for r in results if r == "invalid")
        print(f"[Dashboard] Ván #{self.game_count} | "
              f"Win={wins} Lose={loses} Inv={invs} | "
              f"WinRate={last_wr:.1%} | "
              f"Loss={last_loss:.4f} | "
              f"π_max={last_pi:.4f} | "
              f"Buffer={len(self.buffer):,}")

        fig = plt.figure(figsize=(16, 10))
        fig.suptitle(f"Ván #{self.game_count} | Win: {self._win_rate():.1%} | "
                     f"Buffer: {len(self.buffer):,}", fontsize=14, fontweight="bold")
        gs = gridspec.GridSpec(2, 3, figure=fig, hspace=.4, wspace=.35)

        ax1 = fig.add_subplot(gs[0, 0])
        if losses:
            ax1.plot(l_games, losses, alpha=.25, color="steelblue", lw=.8)
            if len(losses) >= smooth_window:
                ax1.plot(l_games[smooth_window-1:], self._smooth(losses, smooth_window),
                         color="steelblue", lw=2)
        ax1.set_title("Loss"); ax1.grid(alpha=.3)

        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(games, pi_maxs, alpha=.25, color="darkorange", lw=.8)
        if len(pi_maxs) >= smooth_window:
            ax2.plot(games[smooth_window-1:], self._smooth(pi_maxs, smooth_window),
                     color="darkorange", lw=2)
        ax2.set_title("π_max"); ax2.grid(alpha=.3)

        ax3 = fig.add_subplot(gs[0, 2])
        ax3.plot(games, wr, color="seagreen", lw=2)
        ax3.axhline(.5, color="gray", ls="--", lw=1, alpha=.6)
        ax3.set_title(f"Win Rate (rolling {smooth_window})"); ax3.set_ylim(0,1); ax3.grid(alpha=.3)

        ax4 = fig.add_subplot(gs[1, 0:2])
        b=50; nb=max(1,len(results)//b)
        bi,bw,bl,bv,bd=[],[],[],[],[]
        for k in range(nb):
            ch=results[k*b:(k+1)*b]; bi.append(k*b+b//2)
            bw.append(sum(1 for r in ch if r=="win"))
            bl.append(sum(1 for r in ch if r=="lose"))
            bv.append(sum(1 for r in ch if r=="invalid"))
            bd.append(sum(1 for r in ch if r=="draw"))
        ax4.bar(bi,bw,width=b*.8,label="Win",color="seagreen",alpha=.8)
        ax4.bar(bi,bl,width=b*.8,label="Lose",color="tomato",alpha=.8,bottom=bw)
        ax4.bar(bi,bv,width=b*.8,label="Invalid",color="goldenrod",alpha=.8,
                bottom=np.array(bw)+np.array(bl))
        ax4.bar(bi,bd,width=b*.8,label="Draw",color="steelblue",alpha=.5,
                bottom=np.array(bw)+np.array(bl)+np.array(bv))
        ax4.set_title(f"Per {b} games"); ax4.legend(fontsize=9); ax4.grid(alpha=.3,axis="y")

        ax5 = fig.add_subplot(gs[1, 2])
        ax5.plot(games, n_steps, alpha=.3, color="mediumpurple", lw=.8)
        if len(n_steps) >= smooth_window:
            ax5.plot(games[smooth_window-1:], self._smooth(n_steps, smooth_window),
                     color="mediumpurple", lw=2)
        ax5.set_title("Steps/game"); ax5.grid(alpha=.3)
        plt.savefig("dashboard.png", dpi=100, bbox_inches="tight")
        self._show_figure(fig)

    def _show_figure(self, fig, clear_before=False, close_after=True):
        """Hiển thị figure ổn định trong notebook và script."""
        try:
            if get_ipython() is not None:
                if clear_before:
                    display.clear_output(wait=True)
                buf = BytesIO()
                fig.savefig(buf, format="png", bbox_inches="tight")
                buf.seek(0)
                display.display(Image(data=buf.getvalue()))
                if close_after:
                    plt.close(fig)
                return
        except Exception:
            pass

        plt.show()
        if close_after:
            plt.close(fig)

    def plot_last_game_detail(self):
        if not self.game_history: return
        log = self.game_history[-1]
        print(f"Ván #{log['game']} | {log['result'].upper()} | "
              f"steps={log['total_steps']} loss={log['total_loss']}")

    def _smooth(self, values, window=20):
        return np.convolve(values, np.ones(window)/window, mode="valid")
    def _win_rate(self):
        if not self.game_history: return 0.0
        return sum(1 for g in self.game_history if g["result"]=="win") / len(self.game_history)

    # ── SAVE / LOAD ──────────────────────────────────────────────

    def save(self, path="caro_az.keras"):
        self.model.save(path); print(f"Đã lưu → {path}")

    def load(self, path="caro_az.keras"):
        with self.strategy.scope():
            self.model = tf.keras.models.load_model(path)
            _d = np.zeros((1, self.board_size, self.board_size, 3), dtype=np.float32)
            self.model(_d, training=False); self.model(_d, training=True)
            self.optimizer = tf.keras.optimizers.Adam(LR, weight_decay=WEIGHT_DECAY)
            self.optimizer.build(self.model.trainable_variables)
        self.mcts.model = self.model
        print(f"Đã load ← {path}")
