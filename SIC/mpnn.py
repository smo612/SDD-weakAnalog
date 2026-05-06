"""SIC/mpnn.py - Hybrid MP+NN Digital SIC (V6.5 真實訓練版)

V6.5 新增：
- fit_mp_per_sample_and_collect_residuals(): 真正對每個樣本做MP擬合
- NNTrainer: 真實的NN訓練器（dataset-level正規化 + Energy loss）
"""

import numpy as np
import torch
import torch.nn as nn
import logging
from .mp import MPBackend
from .features import build_short_window_features, normalize_features_dataset, build_mp_features
from .utils import compute_ls_alpha, compute_digital_sic_metrics

logger = logging.getLogger(__name__)


class ResidualMLP(nn.Module):
    """小型MLP（殘差學習）"""
    
    def __init__(self, input_dim, hidden_dims=[64, 32], dropout=0.1):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h_dim
        
        layers.append(nn.Linear(prev_dim, 2))  # 輸出 I/Q
        
        self.net = nn.Sequential(*layers)
        
        param_count = sum(p.numel() for p in self.parameters())
        logger.info(f"[ResidualMLP] 參數量: {param_count:,}")
    
    def forward(self, x):
        """
        Args:
            x: [B, D] 或 [N, D]
        Returns:
            out: [B, 2] 或 [N, 2]
        """
        out = self.net(x)
        return out


class NNTrainer:
    """NN訓練器（V6.5：真實訓練）"""
    
    def __init__(self, model, device='cpu', lr=1e-3, epochs=10, 
                 batch_size=2048, lambda_energy=0.5):
        """
        Args:
            model: ResidualMLP模型
            device: 'cpu' or 'cuda'
            lr: 學習率
            epochs: 訓練輪數
            batch_size: 批次大小（符號數）
            lambda_energy: Energy loss權重
        """
        self.model = model
        self.device = device
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.lambda_energy = lambda_energy
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        
        logger.info(f"[NNTrainer] 初始化完成")
        logger.info(f"  學習率: {lr}, Epochs: {epochs}, Batch: {batch_size}")
        logger.info(f"  Energy loss權重: {lambda_energy}")
    
    def train(self, train_features, train_targets, val_features, val_targets):
        """
        訓練NN（V6.5：真實訓練流程）
        
        Args:
            train_features: [N_train, D] float32（已正規化）
            train_targets: [N_train] complex64（MP殘差）
            val_features: [N_val, D] float32（已正規化）
            val_targets: [N_val] complex64
        """
        logger.info("[NNTrainer] 開始訓練...")
        
        # 轉為PyTorch
        X_train = torch.from_numpy(train_features).float().to(self.device)
        y_train = torch.from_numpy(train_targets).to(torch.complex64).to(self.device)
        
        X_val = torch.from_numpy(val_features).float().to(self.device)
        y_val = torch.from_numpy(val_targets).to(torch.complex64).to(self.device)
        
        # ✅ 防呆：確保X和y長度一致
        assert X_train.shape[0] == y_train.shape[0], \
            f"訓練集X/Y長度不符: {X_train.shape[0]} vs {y_train.shape[0]} (請檢查window_L對齊)"
        assert X_val.shape[0] == y_val.shape[0], \
            f"驗證集X/Y長度不符: {X_val.shape[0]} vs {y_val.shape[0]} (請檢查window_L對齊)"
        
        logger.info(f"[NNTrainer] 訓練集: X={X_train.shape}, y={y_train.shape}")
        logger.info(f"[NNTrainer] 驗證集: X={X_val.shape}, y={y_val.shape}")
        
        N_train = X_train.shape[0]
        n_batches = (N_train + self.batch_size - 1) // self.batch_size
        
        best_val_loss = float('inf')
        
        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0.0
            total_mse = 0.0
            total_energy = 0.0
            
            # 隨機打亂
            perm = torch.randperm(N_train)
            X_train_shuffled = X_train[perm]
            y_train_shuffled = y_train[perm]
            
            for batch_idx in range(n_batches):
                start = batch_idx * self.batch_size
                end = min(start + self.batch_size, N_train)
                
                X_batch = X_train_shuffled[start:end]
                y_batch = y_train_shuffled[start:end]
                
                # Forward
                pred = self.model(X_batch)  # [B, 2]
                pred_complex = torch.complex(pred[:, 0], pred[:, 1])
                
                # Loss 1: MSE loss（I/Q分開）
                y_batch_iq = torch.stack([y_batch.real, y_batch.imag], dim=-1)
                loss_mse = self.criterion(pred, y_batch_iq)
                
                # Loss 2: Energy loss（LS-α還原後的殘差能量）
                alpha_num = torch.sum(pred_complex.conj() * y_batch)
                alpha_den = torch.sum(pred_complex.conj() * pred_complex) + 1e-12
                alpha = alpha_num / alpha_den
                
                pred_scaled = alpha * pred_complex
                residual = y_batch - pred_scaled
                
                E_before = torch.mean(torch.abs(y_batch) ** 2)
                E_after = torch.mean(torch.abs(residual) ** 2)
                energy_ratio = E_after / (E_before + 1e-12)
                
                loss_energy = energy_ratio
                
                # Total loss
                loss = loss_mse + self.lambda_energy * loss_energy
                
                # Backward
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                
                total_loss += loss.item()
                total_mse += loss_mse.item()
                total_energy += loss_energy.item()
            
            # Epoch統計
            avg_loss = total_loss / n_batches
            avg_mse = total_mse / n_batches
            avg_energy = total_energy / n_batches
            
            # 驗證
            self.model.eval()
            with torch.no_grad():
                val_pred = self.model(X_val)
                val_pred_complex = torch.complex(val_pred[:, 0], val_pred[:, 1])
                
                # Val MSE
                val_y_iq = torch.stack([y_val.real, y_val.imag], dim=-1)
                val_mse = self.criterion(val_pred, val_y_iq)
                
                # Val Energy
                val_alpha_num = torch.sum(val_pred_complex.conj() * y_val)
                val_alpha_den = torch.sum(val_pred_complex.conj() * val_pred_complex) + 1e-12
                val_alpha = val_alpha_num / val_alpha_den
                
                val_pred_scaled = val_alpha * val_pred_complex
                val_residual = y_val - val_pred_scaled
                
                val_E_before = torch.mean(torch.abs(y_val) ** 2)
                val_E_after = torch.mean(torch.abs(val_residual) ** 2)
                val_energy_ratio = val_E_after / (val_E_before + 1e-12)
                
                # Val Suppression（dB）
                val_supp_db = 10 * torch.log10(val_E_before / (val_E_after + 1e-12))
                
                val_loss = val_mse.item() + self.lambda_energy * val_energy_ratio.item()
            
            logger.info(f"[Epoch {epoch+1}/{self.epochs}] "
                       f"Train Loss={avg_loss:.4f} (MSE={avg_mse:.4f}, E={avg_energy:.4f}) | "
                       f"Val Loss={val_loss:.4f}, Val Supp={val_supp_db.item():.2f} dB")
            
            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                logger.info(f"  ✅ 新的最佳Val Loss: {val_loss:.4f}")
        
        logger.info("[NNTrainer] 訓練完成！")


class MPNNBackend:
    """Hybrid MP+NN Backend (V6.5 真實訓練版)"""
    
    def __init__(self, mp_config, nn_config, device='cpu'):
        """
        初始化MPNN Backend
        
        Args:
            mp_config: MP配置 dict
            nn_config: NN配置 dict {
                'window_L': 13,
                'hidden': [64, 32],
                'dropout': 0.1,
                'lr': 1e-3,
                'epochs': 10,
                'batch_symbols': 2048,
                'lambda_energy': 0.5
            }
            device: 'cpu' or 'cuda'
        """
        # MP部分
        self.mp_backend = MPBackend(**mp_config)
        
        # NN部分
        self.window_L = nn_config.get('window_L', 13)
        self.hidden_dims = nn_config.get('hidden', [64, 32])
        self.dropout = nn_config.get('dropout', 0.1)
        self.lr = nn_config.get('lr', 1e-3)
        self.epochs = nn_config.get('epochs', 10)
        self.batch_symbols = nn_config.get('batch_symbols', 2048)
        self.lambda_energy = nn_config.get('lambda_energy', 0.5)
        
        self.device = device
        
        # 特徵維度：4*L*2 (x, |x|²x, x*, x*|x|² 各L個，實部+虛部)
        self.input_dim = 4 * self.window_L * 2
        
        # 立即建立模型
        self.model = ResidualMLP(
            input_dim=self.input_dim,
            hidden_dims=self.hidden_dims,
            dropout=self.dropout
        ).to(self.device)
        
        # 建立訓練器
        self.trainer = NNTrainer(
            model=self.model,
            device=self.device,
            lr=self.lr,
            epochs=self.epochs,
            batch_size=self.batch_symbols,
            lambda_energy=self.lambda_energy
        )
        
        self.feature_stats = None  # dataset-level正規化統計量
        
        # Gate控制
        self.gate_patience = nn_config.get('gate_patience', 2)
        self.gate_counter = 0
        self.use_gate = True
        
        logger.info(f"[MPNN] 初始化完成")
        logger.info(f"  MP: {mp_config}")
        logger.info(f"  NN窗口: L={self.window_L}, 特徵維度: {self.input_dim}")
        logger.info(f"  NN結構: {self.hidden_dims}")
        logger.info(f"  訓練: epochs={self.epochs}, batch={self.batch_symbols}")
        logger.info(f"  Device: {device}")
    
    def fit_mp_per_sample_and_collect_residuals(self, samples):
        """
        V6.5核心方法：對每個樣本獨立做MP擬合，收集殘差特徵
        
        Args:
            samples: 樣本列表 [{
                'y': [N],
                'x': [N],
                'si_after_analog': [N],
                ...
            }, ...]
        
        Returns:
            features_concat: [N_total, D] float32（已正規化）
            targets_concat: [N_total] complex64（MP殘差）
            mp_suppressions: [n_samples] MP抑制量（dB）
        """
        logger.info(f"[MPNN] 開始per-sample MP擬合（{len(samples)}個樣本）...")
        
        all_features = []
        all_targets = []
        mp_suppressions = []
        
        for i, sample in enumerate(samples):
            # 1. 對此樣本擬合MP
            self.mp_backend.fit(sample)
            
            # 2. 用MP預測
            x = sample['x']
            si_after_analog = sample['si_after_analog']
            
            Phi_mp = build_mp_features(
                x,
                poly_orders=self.mp_backend.poly_orders,
                memory_len=self.mp_backend.memory_len,
                with_conj=self.mp_backend.with_conj
            )
            r_mp_shape = Phi_mp @ self.mp_backend.w_mp
            
            # LS-α
            alpha_mp = compute_ls_alpha(r_mp_shape, si_after_analog)
            r_mp = alpha_mp * r_mp_shape
            
            # 3. 計算MP殘差
            y_res_mp = si_after_analog - r_mp
            
            # 4. 計算MP抑制量
            P_before = np.mean(np.abs(si_after_analog) ** 2)
            P_after = np.mean(np.abs(y_res_mp) ** 2)
            mp_supp = 10 * np.log10(P_before / (P_after + 1e-12))
            mp_suppressions.append(mp_supp)
            
            logger.info(f"  樣本 {i+1}/{len(samples)}: MP抑制 = {mp_supp:.2f} dB, "
                       f"rsi_scale={sample.get('rsi_scale', 'N/A')}")
            
            # 5. 構建NN特徵（基於MP殘差）
            features_sample = build_short_window_features(x, y_res_mp, window_L=self.window_L)
            targets_sample = y_res_mp[self.window_L:].copy()  # 去掉窗口導致的前幾個樣本
            
            # ✅ 關鍵修正：特徵也裁掉前L列，與目標對齊
            features_sample = features_sample[self.window_L:]
            
            # 驗證長度
            assert len(features_sample) == len(targets_sample), \
                f"樣本 {i+1} 長度不匹配！F:{len(features_sample)} vs T:{len(targets_sample)}"
            
            all_features.append(features_sample)
            all_targets.append(targets_sample)
        
        # 6. 拼接所有樣本
        features_concat = np.concatenate(all_features, axis=0)
        targets_concat = np.concatenate(all_targets, axis=0)
        
        logger.info(f"[MPNN] ✅ MP擬合完成")
        logger.info(f"  MP抑制範圍: {min(mp_suppressions):.2f} ~ {max(mp_suppressions):.2f} dB")
        logger.info(f"  拼接後特徵: {features_concat.shape}")
        logger.info(f"  拼接後目標: {targets_concat.shape}")
        
        return features_concat, targets_concat, mp_suppressions
    
    def fit(self, data_dict):
        """
        標準fit介面（單樣本訓練，保持向後兼容）
        
        Args:
            data_dict: {
                'y': [N],
                'x': [N],
                'si_after_analog': [N]
            }
        """
        # === Stage 1: MP ===
        logger.info("[MPNN] Stage 1: MP估計")
        self.mp_backend.fit(data_dict)
        
        # === Stage 2: NN學習殘差 ===
        logger.info("[MPNN] Stage 2: NN殘差學習")
        
        y = data_dict['y']
        x = data_dict['x']
        si_after_analog = data_dict.get('si_after_analog', None)
        
        if si_after_analog is None:
            logger.error("[MPNN] 需要si_after_analog來訓練NN")
            raise ValueError("MPNN訓練需要si_after_analog")
        
        N = len(y)
        
        # MP預測
        Phi_mp = build_mp_features(
            x,
            poly_orders=self.mp_backend.poly_orders,
            memory_len=self.mp_backend.memory_len,
            with_conj=self.mp_backend.with_conj
        )
        r_mp_shape = Phi_mp @ self.mp_backend.w_mp
        alpha_mp = compute_ls_alpha(r_mp_shape, si_after_analog)
        r_mp = alpha_mp * r_mp_shape
        
        # MP殘差
        y_res_mp = si_after_analog - r_mp
        
        logger.info(f"[MPNN] MP殘差功率: {np.mean(np.abs(y_res_mp)**2):.2e}")
        
        # 構建NN特徵
        features = build_short_window_features(x, y_res_mp, window_L=self.window_L)
        
        # ✅ 關鍵修正：特徵也裁掉前L列，與目標對齊
        features = features[self.window_L:]
        
        # 正規化（使用資料集統計）
        features_norm, self.feature_stats = normalize_features_dataset(features, stats=None)
        
        # Target: MP殘差
        target = y_res_mp[self.window_L:].copy()
        
        # 驗證長度
        assert len(features_norm) == len(target), \
            f"特徵/目標長度不匹配！F:{len(features_norm)} vs T:{len(target)}"
        
        # 簡單訓練（無驗證集）
        self.trainer.train(features_norm, target, features_norm[:1000], target[:1000])
        
        logger.info("[MPNN] 訓練完成")
        
        return self
    
    def predict(self, batch_dict):
        """
        預測：MP + NN殘差
        
        Args:
            batch_dict: {
                'y': [N],
                'x': [N],
                'si_after_analog': [N],
                'P_signal': float,
                'P_noise': float
            }
        
        Returns:
            r_hat: [N]
            metrics: dict
        """
        if self.mp_backend.w_mp is None or self.model is None:
            raise RuntimeError("[MPNN] 必須先呼叫fit()")
        
        y = batch_dict['y']
        x = batch_dict['x']
        si_after_analog = batch_dict.get('si_after_analog', None)
        P_signal = batch_dict.get('P_signal', None)
        P_noise = batch_dict.get('P_noise', None)
        
        # === MP預測 ===
        Phi_mp = build_mp_features(
            x,
            poly_orders=self.mp_backend.poly_orders,
            memory_len=self.mp_backend.memory_len,
            with_conj=self.mp_backend.with_conj
        )
        r_mp_shape = Phi_mp @ self.mp_backend.w_mp
        
        if si_after_analog is not None:
            y_res_mp = si_after_analog
        else:
            y_res_mp = y
        
        alpha_mp = compute_ls_alpha(r_mp_shape, y_res_mp)
        r_mp = alpha_mp * r_mp_shape
        
        # === NN預測殘差 ===
        y_res_after_mp = si_after_analog - r_mp if si_after_analog is not None else y - r_mp
        
        features = build_short_window_features(x, y_res_after_mp, window_L=self.window_L)
        
        # ✅ 關鍵修正：特徵裁掉前L列（與訓練時一致）
        features = features[self.window_L:]
        
        features_norm, _ = normalize_features_dataset(features, stats=self.feature_stats)
        
        features_torch = torch.from_numpy(features_norm).float().to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            pred = self.model(features_torch)  # [N-L, 2]
            pred_complex = torch.complex(pred[:, 0], pred[:, 1])
            r_nn_shape = pred_complex.cpu().numpy()
        
        # 補齊窗口長度（前L個設為0）
        r_nn_shape_full = np.zeros(len(x), dtype=np.complex64)
        r_nn_shape_full[self.window_L:] = r_nn_shape
        
        # NN α計算
        alpha_nn = compute_ls_alpha(r_nn_shape_full, y_res_after_mp)
        r_nn = alpha_nn * r_nn_shape_full
        
        # === Gate決策 ===
        if self.use_gate:
            E_mp = np.mean(np.abs(y_res_after_mp) ** 2)
            y_res_after_mpnn = y_res_after_mp - r_nn
            E_mpnn = np.mean(np.abs(y_res_after_mpnn) ** 2)
            
            use_nn = (E_mpnn < E_mp / 1.259)  # 1 dB margin
            
            if use_nn:
                r_hat = r_mp + r_nn
                gate_used = False
                method = 'mpnn'
            else:
                r_hat = r_mp
                gate_used = True
                method = 'mp_fallback'
                logger.warning(f"[MPNN] Gate回退MP: E_mpnn/E_mp = {E_mpnn/E_mp:.3f}")
        else:
            r_hat = r_mp + r_nn
            gate_used = False
            method = 'mpnn'
        
        # 消除SI
        y_clean = y - r_hat
        
        # 計算metrics
        if si_after_analog is not None:
            si_residual = si_after_analog - r_hat
            metrics = compute_digital_sic_metrics(
                y_before=y,
                y_after=y_clean,
                si_before=si_after_analog,
                si_after=si_residual,
                r_hat_scaled=r_hat,
                P_signal=P_signal,
                P_noise=P_noise,
                alpha=alpha_mp
            )
        else:
            metrics = {
                'Digital_supp_si': 0.0,
                'Total_supp_SI_only': 0.0,
                'SINR_after_digital': None,
                'alpha': {
                    'real': float(alpha_mp.real),
                    'imag': float(alpha_mp.imag),
                    'abs': float(np.abs(alpha_mp))
                }
            }
        
        metrics['gate_used'] = gate_used
        metrics['method'] = method
        
        logger.info(f"[MPNN] Digital Supp: {metrics['Digital_supp_si']:.2f} dB ({method})")
        if metrics['SINR_after_digital'] is not None:
            logger.info(f"[MPNN] SINR: {metrics['SINR_after_digital']:.2f} dB")
        
        return r_hat, metrics