import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
from IPython import display
import os
from datetime import datetime

# ================================================================
# CONFIG
# ================================================================
BOARD_SIZE = 40
WIN_LEN    = 5


# ================================================================
# ENVIRONMENT
# ================================================================

class CaroEnv:
    """
    Môi trường chơi caro 40x40.

    3 chế độ:
        play_one(a1, a2)              — chơi 1 ván, hiển thị bàn cờ cuối
        play_n(a1, a2, n)             — chơi n ván, hiển thị ván thắng mỗi phe
        train(a1, a2, n, ckpt_every)  — train n ván, lưu checkpoint, vẽ biểu đồ
    """

    def __init__(self, board_size=BOARD_SIZE):
        self.board_size = board_size
        self.board      = np.zeros((board_size, board_size), dtype=np.float32)
        self.done       = False
        self.winner     = None          # 1, -1, hoặc 0 (hoà)
        self.move_log   = []            # [(player, row, col), ...]

    # ----------------------------------------------------------------
    # CORE
    # ----------------------------------------------------------------

    def reset(self):
        self.board    = np.zeros((self.board_size, self.board_size), dtype=np.float32)
        self.done     = False
        self.winner   = None
        self.move_log = []

    def step(self, player, row, col):
        """
        Đặt quân player (1 hoặc -1) vào ô (row, col).

        Trả về:
            done   : bool   — ván đã kết thúc chưa
            result : str    — "win" / "lose" / "invalid" / "draw" / None
                             (từ góc nhìn của player)
        """
        # Đi sai luật
        if not (0 <= row < self.board_size and 0 <= col < self.board_size):
            self.done   = True
            self.winner = -player
            return True, "invalid"

        if self.board[row, col] != 0:
            self.done   = True
            self.winner = -player
            return True, "invalid"

        # Đặt quân
        self.board[row, col] = player
        self.move_log.append((player, row, col))

        # Kiểm tra thắng
        if self._check_win(player, row, col):
            self.done   = True
            self.winner = player
            return True, "win"

        # Kiểm tra hoà (đầy bàn)
        if np.all(self.board != 0):
            self.done   = True
            self.winner = 0
            return True, "draw"

        return False, None

    def _check_win(self, player, row, col):
        """Kiểm tra player vừa thắng sau nước đi (row, col) chưa."""
        return self._check_win_on_board(self.board, player, row, col)

    def _check_win_on_board(self, board, player, row, col):
        """Kiểm tra thắng trên board bất kỳ (không phụ thuộc self.board)."""
        directions = [(0,1), (1,0), (1,1), (1,-1)]
        for dr, dc in directions:
            count = 1
            for sign in [1, -1]:
                r, c = row + sign*dr, col + sign*dc
                while 0 <= r < self.board_size and 0 <= c < self.board_size \
                      and board[r, c] == player:
                    count += 1
                    r += sign*dr
                    c += sign*dc
            if count >= WIN_LEN:
                return True
        return False

    def _generate_random_opening(self, prefill_steps):
        """
        Sinh trạng thái bàn cờ ngẫu nhiên với prefill_steps nước đã đi sẵn.

        Điều kiện hợp lệ:
          - prefill_steps là số chẵn
          - số quân 2 bên bằng nhau
          - không có trạng thái thắng trong suốt quá trình đặt
        """
        total_cells = self.board_size * self.board_size
        if prefill_steps < 0:
            raise ValueError("prefill_steps phải >= 0")
        if prefill_steps % 2 != 0:
            raise ValueError("prefill_steps phải là số chẵn để 2 bên đi bằng nhau")
        if prefill_steps >= total_cells:
            raise ValueError("prefill_steps phải nhỏ hơn tổng số ô của bàn cờ")

        board = np.zeros((self.board_size, self.board_size), dtype=np.float32)
        move_log = []
        turn = 1

        for _ in range(prefill_steps):
            empties = np.argwhere(board == 0)
            np.random.shuffle(empties)

            placed = False
            for r, c in empties:
                rr, cc = int(r), int(c)
                board[rr, cc] = turn

                if not self._check_win_on_board(board, turn, rr, cc):
                    move_log.append((turn, rr, cc))
                    placed = True
                    break

                board[rr, cc] = 0

            if not placed:
                raise RuntimeError(
                    "Không thể sinh bàn mở đầu hợp lệ. Hãy giảm prefill_steps."
                )

            turn *= -1

        return board, move_log, 1

    def _play_game(self, agent1, agent2,
                   initial_board=None, initial_move_log=None, start_turn=1):
        """
        Chơi 1 ván hoàn chỉnh giữa agent1 (player=1) và agent2 (player=-1).

        Trả về:
            winner  : 1, -1, hoặc 0
            results : dict  — {"agent1": "win"/"lose"/"draw", "agent2": ...}
        """
        if initial_board is None:
            self.reset()
        else:
            self.board = initial_board.copy().astype(np.float32)
            self.done = False
            self.winner = None
            self.move_log = initial_move_log.copy() if initial_move_log is not None else []

        agents  = {1: agent1, -1: agent2}
        turn    = start_turn

        while not self.done:
            agent  = agents[turn]
            row, col = agent.get_action(self.board.copy(), turn)
            done, result = self.step(turn, row, col)

            if done:
                # Xác định kết quả từng phe
                if self.winner == 0:
                    r1, r2 = "draw", "draw"
                elif result == "invalid":
                    r1 = "invalid" if turn == 1  else "win"
                    r2 = "invalid" if turn == -1 else "win"
                else:
                    r1 = "win"  if self.winner == 1  else "lose"
                    r2 = "win"  if self.winner == -1 else "lose"
                break

            turn *= -1   # đổi lượt

        return self.winner, {"agent1": r1, "agent2": r2}

    # ----------------------------------------------------------------
    # CHẾ ĐỘ 1: CHƠI 1 VÁN
    # ----------------------------------------------------------------

    def play_one(self, agent1, agent2, title="", prefill_steps=0, collect=False):
        """
        Chơi 1 ván và hiển thị bàn cờ cuối dưới dạng hình ảnh.
        collect=False: đánh giá, không lưu vào buffer.
        collect=True:  lưu data vào buffer (hiếm khi cần).
        """
        if prefill_steps > 0:
            opening_board, opening_log, start_turn = self._generate_random_opening(prefill_steps)
            winner, results = self._play_game(
                agent1,
                agent2,
                initial_board=opening_board,
                initial_move_log=opening_log,
                start_turn=start_turn,
            )
        else:
            winner, results = self._play_game(agent1, agent2)

        if collect:
            agent1.train_on_game(results["agent1"])
            agent2.train_on_game(results["agent2"])
        else:
            # Xoá _steps tích lũy từ get_action để không lẫn vào lần train sau
            if hasattr(agent1, "_steps"): agent1._steps = []
            if hasattr(agent2, "_steps"): agent2._steps = []

        label = {1: "Agent 1", -1: "Agent 2", 0: "Hoà"}
        print(f"Kết quả: {label.get(winner, '?')} thắng")
        print(f"  Agent1 → {results['agent1']}  |  Agent2 → {results['agent2']}")

        self._draw_board(
            self.board,
            self.move_log,
            title=title or f"Ván đấu — {label.get(winner,'?')} thắng",
        )
        
    def play_live(self, agent1, agent2, title="", delay=0.15, prefill_steps=0, collect=False):
        """
        Chơi 1 ván và cập nhật bàn cờ liên tục từng bước.
        delay: thời gian nghỉ giữa 2 nước (giây).
        """
        import time
        if prefill_steps > 0:
            opening_board, opening_log, start_turn = self._generate_random_opening(prefill_steps)
            self.board = opening_board.copy().astype(np.float32)
            self.done = False
            self.winner = None
            self.move_log = opening_log.copy()
            turn = start_turn
        else:
            self.reset()
            turn = 1

        agents = {1: agent1, -1: agent2}

        fig, ax = plt.subplots(figsize=(8, 8))

        while not self.done:
            agent = agents[turn]
            row, col = agent.get_action(self.board.copy(), turn)
            done, result = self.step(turn, row, col)

            # Vẽ lại sau mỗi nước đi
            ax.clear()
            label = {1: "Agent 1", -1: "Agent 2"}
            self._draw_board_on_ax(
                ax,
                self.board,
                self.move_log,
                title=f"{title} | Turn: {label.get(turn, '?')} | Move: ({row}, {col})"
            )
            display.display(fig)
            display.clear_output(wait=True)
            time.sleep(delay)

            if done:
                if self.winner == 0:
                    r1, r2 = "draw", "draw"
                elif result == "invalid":
                    r1 = "invalid" if turn == 1 else "win"
                    r2 = "invalid" if turn == -1 else "win"
                else:
                    r1 = "win" if self.winner == 1 else "lose"
                    r2 = "win" if self.winner == -1 else "lose"

                if collect:
                    agent1.train_on_game(r1)
                    agent2.train_on_game(r2)
                else:
                    if hasattr(agent1, "_steps"): agent1._steps = []
                    if hasattr(agent2, "_steps"): agent2._steps = []
                break

            turn *= -1

        # Giữ frame cuối
        ax.clear()
        winner_label = {1: "Agent 1", -1: "Agent 2", 0: "Hoà"}
        self._draw_board_on_ax(
            ax,
            self.board,
            self.move_log,
            title=f"Kết quả: {winner_label.get(self.winner, '?')}"
        )
        display.display(fig)
        print(f"Kết quả: {winner_label.get(self.winner, '?')} thắng")
        print(f"Agent1 -> {r1} | Agent2 -> {r2}")

    # ----------------------------------------------------------------
    # CHẾ ĐỘ 2: CHƠI N VÁN
    # ----------------------------------------------------------------

    def play_n(self, agent1, agent2, n=100, collect=False):
        """
        Chơi n ván, hiển thị thống kê + bàn cờ ván thắng đẹp nhất.
        collect=False: đánh giá, không lưu vào buffer.
        """
        stats     = {"agent1": {"win":0,"lose":0,"draw":0,"invalid":0},
                     "agent2": {"win":0,"lose":0,"draw":0,"invalid":0}}
        best_game = {1: None, -1: None}

        for i in range(n):
            winner, results = self._play_game(agent1, agent2)

            if collect:
                agent1.train_on_game(results["agent1"])
                agent2.train_on_game(results["agent2"])
            else:
                if hasattr(agent1, "_steps"): agent1._steps = []
                if hasattr(agent2, "_steps"): agent2._steps = []

            r1, r2 = results["agent1"], results["agent2"]
            stats["agent1"][r1 if r1 in stats["agent1"] else "lose"] += 1
            stats["agent2"][r2 if r2 in stats["agent2"] else "lose"] += 1

            # Lưu ván thắng dài nhất của mỗi phe (dài = đẹp hơn)
            if winner in [1, -1]:
                if best_game[winner] is None or \
                   len(self.move_log) > len(best_game[winner]["moves"]):
                    best_game[winner] = {
                        "board" : self.board.copy(),
                        "moves" : self.move_log.copy(),
                    }

        self._show_play_n_result(stats, best_game, n)

    def _show_play_n_result(self, stats, best_game, n):
        """Hiển thị kết quả n ván."""
        # ── Text log (luôn hiện trong Kaggle logs) ──
        a1, a2 = stats["agent1"], stats["agent2"]
        print(f"[play_n] {n} ván:")
        print(f"  Agent1: W={a1['win']} L={a1['lose']} D={a1['draw']} I={a1['invalid']}"
              f"  ({a1['win']/max(n,1):.0%} win)")
        print(f"  Agent2: W={a2['win']} L={a2['lose']} D={a2['draw']} I={a2['invalid']}"
              f"  ({a2['win']/max(n,1):.0%} win)")

        # ── Chart ──
        fig = plt.figure(figsize=(18, 8))
        fig.suptitle(f"Kết quả {n} ván đấu", fontsize=14, fontweight="bold")
        gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)

        # ── Biểu đồ thống kê ──────────────────────────────────────
        ax_stats = fig.add_subplot(gs[0, 0])
        labels   = ["Win", "Lose", "Draw", "Invalid"]
        colors   = ["seagreen", "tomato", "steelblue", "goldenrod"]
        x        = np.arange(len(labels))
        w        = 0.35

        a1_vals = [stats["agent1"].get(k.lower(), 0) for k in labels]
        a2_vals = [stats["agent2"].get(k.lower(), 0) for k in labels]

        ax_stats.bar(x - w/2, a1_vals, w, label="Agent 1", color=colors, alpha=0.85)
        ax_stats.bar(x + w/2, a2_vals, w, label="Agent 2", color=colors, alpha=0.45)
        ax_stats.set_xticks(x)
        ax_stats.set_xticklabels(labels)
        ax_stats.set_title("Thống kê thắng/thua")
        ax_stats.legend()
        ax_stats.grid(alpha=0.3, axis="y")

        # ── Bàn cờ ván thắng Agent 1 ──────────────────────────────
        ax1 = fig.add_subplot(gs[0, 1])
        if best_game[1]:
            self._draw_board_on_ax(ax1, best_game[1]["board"],
                                   best_game[1]["moves"], "Ván thắng đẹp nhất — Agent 1")
        else:
            ax1.set_title("Agent 1 chưa thắng ván nào")
            ax1.axis("off")

        # ── Bàn cờ ván thắng Agent 2 ──────────────────────────────
        ax2 = fig.add_subplot(gs[0, 2])
        if best_game[-1]:
            self._draw_board_on_ax(ax2, best_game[-1]["board"],
                                   best_game[-1]["moves"], "Ván thắng đẹp nhất — Agent 2")
        else:
            ax2.set_title("Agent 2 chưa thắng ván nào")
            ax2.axis("off")

        plt.show()

    # ----------------------------------------------------------------
    # CHẾ ĐỘ 3: TRAIN
    # ----------------------------------------------------------------

    def train(self, agent1, agent2, n=1000, ckpt_every=200,
              plot_every=100, ckpt_dir="checkpoints", prefill_steps=0):
        """
        Train n ván, tự động:
          - Lưu checkpoint mỗi ckpt_every ván
          - Vẽ biểu đồ mỗi plot_every ván (dùng plot_training của agent)
          - Có thể random sẵn prefill_steps nước mở đầu hợp lệ

        Ví dụ:
            env.train(deep_agent, greedy_agent, n=2000, ckpt_every=500, prefill_steps=20)
        """
        os.makedirs(ckpt_dir, exist_ok=True)
        stats = {"agent1": {"win":0,"lose":0,"draw":0,"invalid":0},
                 "agent2": {"win":0,"lose":0,"draw":0,"invalid":0}}

        if prefill_steps % 2 != 0:
            raise ValueError("prefill_steps phải là số chẵn để 2 bên đi bằng nhau")

        for i in range(1, n + 1):
            if prefill_steps > 0:
                opening_board, opening_log, start_turn = self._generate_random_opening(prefill_steps)
                print(f"[Opening] Ván {i}/{n}: đã tạo xong bàn mở đầu với {prefill_steps} nước.")
                winner, results = self._play_game(
                    agent1,
                    agent2,
                    initial_board=opening_board,
                    initial_move_log=opening_log,
                    start_turn=start_turn,
                )
            else:
                winner, results = self._play_game(agent1, agent2)

            agent1.train_on_game(results["agent1"])
            agent2.train_on_game(results["agent2"])

            r1, r2 = results["agent1"], results["agent2"]
            stats["agent1"][r1 if r1 in stats["agent1"] else "lose"] += 1
            stats["agent2"][r2 if r2 in stats["agent2"] else "lose"] += 1

            # Vẽ biểu đồ
            if i % plot_every == 0:
                display.clear_output(wait=True)
                print(f"  Ván {i}/{n}  |  "
                      f"Agent1: {stats['agent1']}  |  Agent2: {stats['agent2']}")
                # Gọi plot của agent nào có (deep agent)
                if hasattr(agent1, "plot_training") and callable(agent1.plot_training):
                    try:
                        agent1.plot_training()
                    except Exception:
                        pass

            # Lưu checkpoint
            if i % ckpt_every == 0:
                ts    = datetime.now().strftime("%Y%m%d_%H%M%S")
                path1 = os.path.join(ckpt_dir, f"agent1_van{i}_{ts}.keras")
                path2 = os.path.join(ckpt_dir, f"agent2_van{i}_{ts}.keras")
                if hasattr(agent1, "save") and callable(agent1.save):
                    try:
                        agent1.save(path1)
                    except Exception:
                        pass
                if hasattr(agent2, "save") and callable(agent2.save):
                    try:
                        agent2.save(path2)
                    except Exception:
                        pass
                print(f"  ✓ Checkpoint lưu tại ván {i}")

        # Vẽ lần cuối sau khi train xong
        display.clear_output(wait=True)
        print(f"Train xong {n} ván!")
        print(f"  Agent1: {stats['agent1']}")
        print(f"  Agent2: {stats['agent2']}")
        if hasattr(agent1, "plot_training") and callable(agent1.plot_training):
            try:
                agent1.plot_training()
            except Exception:
                pass

    def train_with_buffer(self, agent1, agent2, n=1000,
                          train_every=20, batch_size=256, epochs=3,
                          ckpt_every=200, plot_every=100, ckpt_dir="checkpoints"):
        """
        Train với replay buffer.

        Quy trình:
          - Chơi train_every ván → data vào buffer
          - Shuffle buffer → chia batch → train epochs lượt
          - Lặp lại

        Args:
            agent1, agent2 : 2 agents
            n              : tổng số ván
            train_every    : chơi bao nhiêu ván rồi train 1 lần
            batch_size     : batch size khi train từ buffer
            epochs         : số epochs mỗi lần train
            ckpt_every     : lưu checkpoint mỗi N ván
            plot_every     : vẽ biểu đồ mỗi N ván
            ckpt_dir       : thư mục lưu checkpoint

        Ví dụ:
            env.train_with_buffer(
                agent, greedy, n=500,
                train_every=20, batch_size=256, epochs=3,
            )
        """
        import time
        os.makedirs(ckpt_dir, exist_ok=True)
        stats = {"agent1": {"win":0,"lose":0,"draw":0,"invalid":0},
                 "agent2": {"win":0,"lose":0,"draw":0,"invalid":0}}
        t_start = time.time()

        for i in range(1, n + 1):
            winner, results = self._play_game(agent1, agent2)

            # Lưu data vào buffer (không train ngay)
            agent1.train_on_game(results["agent1"])
            agent2.train_on_game(results["agent2"])

            r1 = results["agent1"]
            stats["agent1"][r1 if r1 in stats["agent1"] else "lose"] += 1
            r2 = results["agent2"]
            stats["agent2"][r2 if r2 in stats["agent2"] else "lose"] += 1

            # Train từ buffer mỗi train_every ván
            if i % train_every == 0:
                loss1, loss2 = 0.0, 0.0
                if hasattr(agent1, "train_buffer") and callable(agent1.train_buffer):
                    loss1 = agent1.train_buffer(batch_size, epochs)
                if hasattr(agent2, "train_buffer") and callable(agent2.train_buffer):
                    loss2 = agent2.train_buffer(batch_size, epochs)
                elapsed = time.time() - t_start
                buf1 = len(agent1.buffer) if hasattr(agent1, "buffer") else 0
                buf2 = len(agent2.buffer) if hasattr(agent2, "buffer") else 0
                print(f"  Ván {i}/{n} | loss1={loss1:.4f} loss2={loss2:.4f} | "
                      f"buf={buf1:,}+{buf2:,} | "
                      f"A1: W={stats['agent1']['win']} "
                      f"L={stats['agent1']['lose']} | {elapsed:.0f}s")

            # Vẽ biểu đồ
            if i % plot_every == 0:
                if hasattr(agent1, "plot_training") and callable(agent1.plot_training):
                    try:
                        agent1.plot_training()
                    except Exception:
                        pass

            # Checkpoint
            if i % ckpt_every == 0:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                if hasattr(agent1, "save") and callable(agent1.save):
                    try:
                        agent1.save(os.path.join(ckpt_dir, f"agent1_v{i}_{ts}.keras"))
                    except Exception:
                        pass
                if hasattr(agent2, "save") and callable(agent2.save):
                    try:
                        agent2.save(os.path.join(ckpt_dir, f"agent2_v{i}_{ts}.keras"))
                    except Exception:
                        pass
                print(f"  ✓ Checkpoint ván {i}")

        # Kết thúc
        display.clear_output(wait=True)
        print(f"Train xong {n} ván!")
        print(f"  Agent1: {stats['agent1']}")
        print(f"  Agent2: {stats['agent2']}")
        if hasattr(agent1, "plot_training") and callable(agent1.plot_training):
            try:
                agent1.plot_training()
            except Exception:
                pass

    # ----------------------------------------------------------------
    # VẼ BÀN CỜ
    # ----------------------------------------------------------------

    def _draw_board(self, board, move_log, title=""):
        """Vẽ bàn cờ độc lập (dùng cho play_one)."""
        fig, ax = plt.subplots(figsize=(8, 8))
        self._draw_board_on_ax(ax, board, move_log, title)
        plt.tight_layout()
        plt.show()

    def _draw_board_on_ax(self, ax, board, move_log, title=""):
        """Vẽ bàn cờ lên ax cho trước."""
        n   = self.board_size
        ax.set_xlim(-0.5, n - 0.5)
        ax.set_ylim(-0.5, n - 0.5)
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.axis("off")

        # Nền bàn cờ
        ax.set_facecolor("#F0D9B5")
        bg = patches.Rectangle((-0.5, -0.5), n, n,
                                color="#F0D9B5", zorder=0)
        ax.add_patch(bg)

        # Lưới
        for i in range(n):
            ax.plot([i, i], [0, n-1], color="#9B7A4A", linewidth=0.4, zorder=1)
            ax.plot([0, n-1], [i, i], color="#9B7A4A", linewidth=0.4, zorder=1)

        # Quân cờ
        last_move = move_log[-1] if move_log else None
        for player, r, c in move_log:
            color      = "black" if player == 1 else "white"
            edge_color = "white" if player == 1 else "black"
            zorder     = 3

            circle = plt.Circle(
                (c, n - 1 - r), 0.42,
                color=color, ec=edge_color, linewidth=1.2, zorder=zorder
            )
            ax.add_patch(circle)

            # Đánh dấu nước đi cuối
            if last_move and (player, r, c) == last_move:
                dot = plt.Circle(
                    (c, n - 1 - r), 0.12,
                    color="red", zorder=4
                )
                ax.add_patch(dot)

        # Hiển thị số thứ tự nước đi (chỉ 10 nước cuối để không rối)
        for idx, (player, r, c) in enumerate(move_log[-10:], start=max(1, len(move_log)-9)):
            text_color = "white" if player == 1 else "black"
            ax.text(c, n - 1 - r, str(idx),
                    ha="center", va="center",
                    fontsize=5, color=text_color, fontweight="bold", zorder=5)


# ================================================================
# DEMO NHANH
# ================================================================

if __name__ == "__main__":

    # Import 2 agent (giả lập bằng random agent để test)
    class RandomAgent:
        def __init__(self):
            self.game_count   = 0
            self.game_history = []

        def get_action(self, board, player):
            empty = [(r, c) for r in range(40) for c in range(40) if board[r, c] == 0]
            idx   = np.random.randint(len(empty))
            return empty[idx]

        def train_on_game(self, result):
            self.game_count += 1

        def plot_training(self): pass
        def save(self, path): pass
        def load(self, path): pass

    env = CaroEnv()
    a1  = RandomAgent()
    a2  = RandomAgent()

    # Thử play_one
    env.play_one(a1, a2, title="Test 1 ván")

    # Thử play_n
    env.play_n(a1, a2, n=20)
