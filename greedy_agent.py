import numpy as np

# ================================================================
# CONFIG
# ================================================================
BOARD_SIZE = 40
A          = 8      # base của công thức a^n
WIN_LEN    = 5      # số quân liên tiếp để thắng

# Các hướng duyệt: (dr, dc)
DIRECTIONS = [
    (0,  1),   # ngang
    (1,  0),   # dọc
    (1,  1),   # chéo xuống phải
    (1, -1),   # chéo xuống trái
]


# ================================================================
# GREEDY AGENT
# ================================================================

class GreedyAgent:
    """
    Agent tham lam — không dùng deep learning.
    Chiến thuật: với mỗi ô trống, tính tổng điểm dựa trên
    tất cả cửa sổ 5 ô liên tiếp đè lên ô đó, rồi chọn ô cao điểm nhất.

    Công thức điểm mỗi cửa sổ:
      - Có cả 2 loại quân          → +0
      - Chỉ có n quân cùng loại    → +a^n  (+n*a thêm nếu là quân ta)
      - Có đúng 4 quân ta          → chọn ngay (thắng luôn)
      - Có đúng 4 quân địch        → chọn ngay (chặn ngay)

    Interface giống CaroAgent để cắm vào môi trường dễ dàng.
    """

    def __init__(self, board_size=BOARD_SIZE, a=A):
        self.board_size = board_size
        self.a          = a

        # Giữ nguyên interface — không dùng nhưng để môi trường gọi được
        self.game_count   = 0
        self.game_history = []

    # ----------------------------------------------------------------
    # INTERFACE CHÍNH — môi trường gọi 2 hàm này
    # ----------------------------------------------------------------

    def get_action(self, board, player):
        """
        board  : np.array (40, 40)  —  1=quân player1, -1=quân player2, 0=trống
        player : 1 hoặc -1

        Trả về : (row, col)
        """
        # Chuẩn hoá: quân ta = 1, quân địch = -1
        state = board * player

        best_score = -1
        best_cells = []

        for r in range(self.board_size):
            for c in range(self.board_size):

                if state[r, c] != 0:
                    continue  # ô đã có quân, bỏ qua

                score, instant_pick = self._score_cell(state, r, c)

                if instant_pick:
                    return r, c   # 4 quân ta hoặc địch → chọn ngay

                if score > best_score:
                    best_score = score
                    best_cells = [(r, c)]    # reset danh sách
                elif score == best_score:
                    best_cells.append((r, c))  # cùng điểm thì gom vào

        # Fallback: nếu không tìm được (không nên xảy ra)
        if not best_cells:
            best_cells = [(r, c) for r in range(self.board_size)
                                  for c in range(self.board_size)
                                  if state[r, c] == 0]
        if not best_cells:
            return (0, 0)

        return best_cells[np.random.randint(len(best_cells))]

    def train_on_game(self, result):
        """Không train — agent tham lam dùng luật cứng."""
        self.game_count += 1
        self.game_history.append({
            "game"        : self.game_count,
            "result"      : result,
            "total_steps" : None,
            "total_loss"  : None,
            "avg_prob"    : None,
        })

    # ----------------------------------------------------------------
    # THUẬT TOÁN THAM LAM
    # ----------------------------------------------------------------

    def _score_cell(self, state, r, c):
        """
        Tính tổng điểm cho ô (r, c) đang trống.

        Trả về:
            score        : float  — tổng điểm
            instant_pick : bool   — True nếu nên chọn ngay (4 quân ta/địch)
        """
        score = 0.0

        for dr, dc in DIRECTIONS:
            # Duyệt tất cả cửa sổ WIN_LEN ô có đè lên (r, c)
            # offset = vị trí của (r,c) trong cửa sổ, từ 0 đến WIN_LEN-1
            for offset in range(WIN_LEN):

                # Ô bắt đầu của cửa sổ
                r0 = r - offset * dr
                c0 = c - offset * dc

                # Ô kết thúc của cửa sổ
                r1 = r0 + (WIN_LEN - 1) * dr
                c1 = c0 + (WIN_LEN - 1) * dc

                # Kiểm tra cửa sổ có nằm trong bàn không
                if not (0 <= r0 < self.board_size and 0 <= c0 < self.board_size and
                        0 <= r1 < self.board_size and 0 <= c1 < self.board_size):
                    continue

                # Lấy 5 ô trong cửa sổ (trừ ô đang xét vì nó đang trống)
                window = [
                    state[r0 + i * dr, c0 + i * dc]
                    for i in range(WIN_LEN)
                    if not (r0 + i * dr == r and c0 + i * dc == c)
                ]

                s, instant = self._score_window(window)

                if instant:
                    return score, True   # chọn ngay, không cần tính tiếp

                score += s

        return score, False

    def _score_window(self, window):
        """
        Tính điểm cho 1 cửa sổ (đã bỏ ô đang xét — còn WIN_LEN-1 = 4 ô).

        window : list 4 giá trị trong {1, -1, 0}
                 1  = quân ta
                -1  = quân địch
                 0  = trống

        Trả về:
            score        : float
            instant_pick : bool
        """
        my_count    = window.count(1)
        enemy_count = window.count(-1)

        # Có cả 2 loại quân → cửa sổ này chết, bỏ qua
        if my_count > 0 and enemy_count > 0:
            return 0.0, False

        # 4 quân ta trong cửa sổ → cộng ô này = thắng ngay
        if my_count == WIN_LEN - 1:
            return 0.0, True

        # 4 quân địch trong cửa sổ → chặn ngay
        if enemy_count == WIN_LEN - 1:
            return 0.0, True

        # Chỉ có quân ta (n quân)
        if my_count > 0:
            n     = my_count
            score = self.a ** n + n * self.a   # a^n + bonus tấn công
            return score, False

        # Chỉ có quân địch (n quân)
        if enemy_count > 0:
            n     = enemy_count
            score = self.a ** n                # a^n, không có bonus
            return score, False

        # Cửa sổ toàn trống
        return 0.0, False

    # ----------------------------------------------------------------
    # PHƯƠNG THỨC KHÔNG DÙNG — giữ để đồng bộ interface
    # ----------------------------------------------------------------

    def plot_training(self, smooth_window=20):
        pass

    def plot_last_game_detail(self):
        pass

    def save(self, path=None):
        pass

    def load(self, path=None):
        pass


# ================================================================
# DEMO NHANH
# ================================================================

if __name__ == "__main__":
    agent = GreedyAgent()

    board       = np.zeros((BOARD_SIZE, BOARD_SIZE))
    board[20, 20] = 1
    board[20, 21] = 1
    board[20, 22] = 1
    board[20, 23] = 1   # 4 quân ta liên tiếp → agent phải chọn [20,24] hoặc [20,19]

    row, col = agent.get_action(board, player=1)
    print(f"Agent chọn ô: ({row}, {col})")   # kỳ vọng: (20, 24) hoặc (20, 19)

    agent.train_on_game("win")
    print(f"game_count: {agent.game_count}")
