# config.py ??FINAL (analog-only, with realistic analog mismatch)
SEED = 2025

# Experiment operating point
SNR_DB = 25.0
SIC_DB_FIXED = 23.0      # target analog suppression (dB), mean
SIC_JITTER_DB = 0.20     # per-run random jitter (dB std) to avoid "too perfect" numbers
NUM_SYMS = 2048

# Batch/report
TARGET_DB = 5.5          # target RSI/Noise after analog (dB)
N_RUNS = 50

# RSI channel (reasonable impairments)
RSI_NUM_TAPS = 5
RSI_IQ_AMBAL = 0.02
RSI_IQ_PHERR_DEG = 2.0
RSI_RAPP_P = 2.0
RSI_RAPP_Asat = 3.0

# Analog canceller imperfections (small, realistic)
ANA_GAIN_ERR_STD = 0.02        # 2% gain sigma
ANA_PHASE_ERR_STD_DEG = 1.0    # 1 degree sigma

# Initial guess; script will calibrate to hit TARGET_DB
#RSI_SCALE = 107900.0
RSI_SCALE = 107900.0              # ?孵??身??
USE_RANDOM_START = False     # ???冽?
FIXED_START_IDX = 0          # ?箏?韏琿?


AUX_DISABLE_IQPA = False          # ???post-PA ?◤ aux chain ?銝甈?
AUX_ASAT_FACTOR = 8.0  # ?嚗? Aux-TX PA ???刻?蝺抒????(High Back-off)
IQ_IMBALANCE = 0.02    # 2% ?′擃?IQ 隤文榆

ASIC_SAFETY_SIGN_FLIP = True     # 雿?flip ??瑁???model 鋆∩耨??corr<0 ?蕃
ASIC_FORCE_LINEAR_WHEN_PA_OFF = True
ASIC_EST_SNR_DB = 100.0
ASIC_NSYM = 2000
ASIC_P = 7
ASIC_L = 9  # 蝣箔?憭扳 RSI_NUM_TAPS = 5
