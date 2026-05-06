"""SIC/api.py - 統一介面

提供 load_backend() 函數，返回統一的 fit()/predict() 介面
"""

import logging
from .wlls import WLLSBackend
from .mp import MPBackend, BlockWiseMPBackend
from .mpnn import MPNNBackend

logger = logging.getLogger(__name__)


def load_backend(name, config):
    """
    載入 Digital SIC Backend
    
    Args:
        name: 'wlls' | 'mp' | 'mpnn'
        config: 配置字典
    
    Returns:
        backend: 具有 fit() 和 predict() 方法的物件
    
    Example:
        >>> config = {
        ...     'mp': {'poly_orders': [1, 3], 'memory_len': 5},
        ...     'mpnn': {'window_L': 13, 'hidden': [64, 32]}
        ... }
        >>> sic = load_backend('mp', config)
        >>> sic.fit(data)
        >>> r_hat, metrics = sic.predict(batch)
    """
    name = name.lower()
    
    if name == 'wlls':
        logger.info("[API] 載入 WLLS Backend")
        wlls_cfg = config.get('wlls', {})
        backend = WLLSBackend(
            L=wlls_cfg.get('L', 5),
            lambda_reg=wlls_cfg.get('lambda_reg', 0.01),
            use_widely_linear=wlls_cfg.get('use_widely_linear', False)
        )
    
    elif name == 'mp':
        logger.info("[API] 載入 MP Backend")
        mp_cfg = config.get('mp', {})
        
        # 選擇 MP 變體
        use_blockwise = mp_cfg.get('use_blockwise', False)
        
        if use_blockwise:
            backend = BlockWiseMPBackend(
                poly_orders=mp_cfg.get('poly_orders', [1, 3]),
                memory_len=mp_cfg.get('memory_len', 5),
                ridge_lambda=mp_cfg.get('ridge_lambda', 1e-3),
                block_size=mp_cfg.get('block_size', 4096),
                update_stride=mp_cfg.get('update_stride', 2048),
                with_conj=mp_cfg.get('with_conj', True)
            )
        else:
            backend = MPBackend(
                poly_orders=mp_cfg.get('poly_orders', [1, 3]),
                memory_len=mp_cfg.get('memory_len', 5),
                ridge_lambda=mp_cfg.get('ridge_lambda', 1e-3),
                block_size=mp_cfg.get('block_size', 4096),
                update_stride=mp_cfg.get('update_stride', 2048),
                with_conj=mp_cfg.get('with_conj', True)
            )
    
    elif name == 'mpnn':
        logger.info("[API] 載入 MPNN Backend")
        mp_cfg = config.get('mp', {})
        mpnn_cfg = config.get('mpnn', {})
        
        backend = MPNNBackend(
            mp_config={
                'poly_orders': mp_cfg.get('poly_orders', [1, 3]),
                'memory_len': mp_cfg.get('memory_len', 5),
                'ridge_lambda': mp_cfg.get('ridge_lambda', 1e-3),
                'block_size': mp_cfg.get('block_size', 4096),
                'update_stride': mp_cfg.get('update_stride', 2048),
                'with_conj': mp_cfg.get('with_conj', True)
            },
            nn_config=mpnn_cfg,
            device=config.get('device', 'cpu')
        )
    
    else:
        raise ValueError(f"Unknown backend: {name}. Choose from ['wlls', 'mp', 'mpnn']")
    
    return backend


def validate_backend_interface(backend):
    """
    驗證 backend 是否實現了必要的介面
    
    Args:
        backend: Backend 物件
    
    Returns:
        is_valid: bool
    """
    required_methods = ['fit', 'predict']
    
    for method in required_methods:
        if not hasattr(backend, method):
            logger.error(f"Backend 缺少必要方法: {method}")
            return False
        
        if not callable(getattr(backend, method)):
            logger.error(f"Backend.{method} 不可呼叫")
            return False
    
    logger.info("[API] Backend 介面驗證通過")
    return True