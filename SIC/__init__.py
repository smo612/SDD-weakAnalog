"""SIC - Digital Self-Interference Cancellation Module

統一的 Digital SIC 介面，支援：
- WLLS（線性基準）
- MP（Memory Polynomial）
- MPNN（Hybrid MP+NN）

使用方式：
    from SIC import load_backend
    
    config = {...}
    sic = load_backend('mp', config)
    sic.fit(data)
    r_hat, metrics = sic.predict(batch)
"""

from .api import load_backend, validate_backend_interface
from .utils import (
    compute_ls_alpha,
    compute_power,
    compute_suppression_db,
    compute_sinr_db,
    compute_digital_sic_metrics
)

__version__ = "1.0.0"
__all__ = [
    'load_backend',
    'validate_backend_interface',
    'compute_ls_alpha',
    'compute_power',
    'compute_suppression_db',
    'compute_sinr_db',
    'compute_digital_sic_metrics'
]