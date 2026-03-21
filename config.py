BOARD_SIZE    = 40
WIN_LEN       = 5

LR            = 1e-3
WEIGHT_DECAY  = 1e-4
C_PUCT        = 1.9
N_SIM         = 50
N_PARALLEL    = 8
DIR_ALPHA     = 0.08
DIR_WEIGHT    = 0.15

# Replay buffer
BUFFER_SIZE   = 50000    # số positions tối đa trong buffer
TRAIN_EVERY   = 20       # train mỗi N ván
TRAIN_BATCH   = 256      # batch size khi train từ buffer
TRAIN_EPOCHS  = 3        # số epochs mỗi lần train

GREEDY_A      = 8
