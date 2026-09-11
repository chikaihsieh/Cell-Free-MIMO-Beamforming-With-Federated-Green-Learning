# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 03:18:52 2026

@author: CKAI
"""
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import layers, models, Model
import matplotlib.pyplot as plt
import xgboost as xg

class AP:
    def __init__(self):

        self.X_train = None
        self.X_val = None
        self.X_test = None

        self.y_train = None
        self.y_val = None
        self.y_test = None

        # module outputs
        self.Z1 = None
        self.Z2 = None
        self.all_A1_by_np = None
        self.all_A2 = None
        self.scaler = StandardScaler()
        self.selected_features = None
        self.model = None
        self.bst = None
    ########################################
    # Load dataset
    ########################################
    def load_dataset_train(self, X, y):
        self.X_train = X
        self.y_train = y
    def load_dataset_test(self, X, y):
        self.X_test = X
        self.y_test = y
    def load_dataset_val(self, X, y):
        self.X_val = X
        self.y_val = y        
    ########################################
    # Block Power Method (PCA)
    ########################################
    def _block_power_method(self, C, num_vecs, num_iter):
        N = C.shape[0]
    
        # initial Q (random orthogonal)
        Q = np.random.randn(N, num_vecs) + 1j*np.random.randn(N, num_vecs)
        Q, _ = np.linalg.qr(Q)
    
        for _ in range(num_iter):
            Z = C @ Q
            Q, R = np.linalg.qr(Z)
    
        # compute corresponding eigenvalues (absolute values)
        # lambda_i = q_i^H C q_i
        eigenvalues = np.array([np.abs(Q[:, i].conj().T @ C @ Q[:, i]) for i in range(num_vecs)])

        return Q, eigenvalues
    def _block_power_method_fedavg(
        self,
        seg_ac,
        client_ranges,
        num_vecs,
        num_iter
    ):
        """
        Federated Block Power Method
    
        Parameters
        ----------
        seg_ac :
            total AC data
            shape = (num_samples, N1)
    
        client_ranges :
            example:
            [(0,100), (100,200), (200,800)]
    
        num_vecs :
            number of eigenvectors
    
        num_iter :
            power iterations
        """
    
        # =====================================================
        # mean removal
        # =====================================================
        seg_mean = np.mean(
            seg_ac,
            axis=0,
            keepdims=True
        )
    
        X = seg_ac - seg_mean
    
        # =====================================================
        # dimension
        # =====================================================
        N = X.shape[1]
    
        # =====================================================
        # initialize Q
        # =====================================================
        Q = (
            np.random.randn(N, num_vecs)
            + 1j*np.random.randn(N, num_vecs)
        )
    
        Q, _ = np.linalg.qr(Q)
    
        total_samples = X.shape[0]
    
        # =====================================================
        # power iterations
        # =====================================================
        for it in range(num_iter):
    
            Z_global = np.zeros(
                (N, num_vecs),
                dtype=np.complex128
            )
    
            # -------------------------------------------------
            # each client local update
            # -------------------------------------------------
            for start_idx, end_idx in client_ranges:
    
                # local data
                X_k = X[start_idx:end_idx]
    
                n_k = X_k.shape[0]
    
                # =============================================
                # local contribution
                # =============================================
                # C_k Q = X_k^H X_k Q
                # without explicitly building covariance
                # =============================================
                Z_k = (
                    X_k.conj().T
                    @
                    (X_k @ Q)
                ) / n_k
    
                # =============================================
                # FedAvg aggregation
                # =============================================
                Z_global += (
                    n_k / total_samples
                ) * Z_k
    
            # -------------------------------------------------
            # server QR
            # -------------------------------------------------
            Q, R = np.linalg.qr(Z_global)
    
        # =====================================================
        # approximate eigenvalues
        # =====================================================
        eigenvalues = np.linalg.norm(
            Z_global,
            axis=0
        )

        return Q, eigenvalues
    def _block_power_method_fedavg_hetero(
    self,
    seg_ac,
    client_ranges,
    client_ratios,
    num_vecs,
    num_iter
    ):
    
        # =====================================================
        # mean removal
        # =====================================================
        seg_mean = np.mean(
            seg_ac,
            axis=0,
            keepdims=True
        )
    
        X = seg_ac - seg_mean
    
        N = X.shape[1]
    
        # =====================================================
        # initialize global Q
        # =====================================================
        Q_global = (
            np.random.randn(N, num_vecs)
            + 1j*np.random.randn(N, num_vecs)
        )
    
        Q_global, _ = np.linalg.qr(Q_global)
    
        total_samples = X.shape[0]
    
        # =====================================================
        # iterations
        # =====================================================
        for it in range(num_iter):
    
            Z_global = np.zeros(
                (N, num_vecs),
                dtype=np.complex128
            )
    
            # -------------------------------------------------
            # each client
            # -------------------------------------------------
            for client_idx, (start_idx, end_idx) in enumerate(client_ranges):
    
                X_k = X[start_idx:end_idx]
    
                n_k = X_k.shape[0]
    
                # =============================================
                # local rank
                # =============================================
                local_rank = max(
                    1,
                    int(num_vecs * client_ratios[client_idx])
                )
    
                # local subspace
                Q_local = Q_global[:, :local_rank]
    
                # =============================================
                # local update
                # =============================================
                Z_local = (
                    X_k.conj().T
                    @
                    (X_k @ Q_local)
                ) / n_k
    
                # =============================================
                # zero padding
                # =============================================
                Z_pad = np.zeros(
                    (N, num_vecs),
                    dtype=np.complex128
                )
    
                Z_pad[:, :local_rank] = Z_local
    
                # =============================================
                # FedAvg
                # =============================================
                Z_global += (
                    n_k / total_samples
                ) * Z_pad
    
            # -------------------------------------------------
            # global QR
            # -------------------------------------------------
            Q_global, _ = np.linalg.qr(Z_global)
    
        eigenvalues = np.linalg.norm(
            Z_global,
            axis=0
        )
    
        return Q_global, eigenvalues
    ########################################
    # Module 1: PCA with segmentation
    ########################################
    def complex_to_real_imag(self, X):
            """
            將複數矩陣拆成實部與虛部
            Input:
                X: np.ndarray, shape (num_samples, num_features), dtype=complex
            Output:
                X_real: np.ndarray, shape (num_samples, 2*num_features), dtype=float
                         前半是實部，後半是虛部
            """
            if not np.iscomplexobj(X):
                raise ValueError("Input must be complex array")
        
            X_real = np.hstack([X.real, X.imag])
            return X_real
    def module1_operator(self, use_module=True, N1=4, N2=4, num_iter=10):
            if not use_module:
                self.Z1 = self.X_train.reshape(self.X_train.shape[0], -1)
                return self.Z1
        
            # X shape: (B, 1, Nt, Np)
            B, _, Nt, Np = self.X_train.shape
            X_k = self.X_train[:, 0, :, :] # (B, Nt, Np)
            L1 = Nt // N1
            L2 = Np // N2
            
            ########################################
            # Stage 1: Nt 維度處理 (Spatial)
            ########################################
            a1_dc_unit = np.ones((N1, 1)) / np.sqrt(N1)
            
            # 建立一個容器來存每個 np 的 Anchors
            # 最終我們需要為每個 np 找到 Nt 個 Anchors
            self.all_A1_by_np = [] 
    
            for np_idx in range(Np):
                all_q1_vectors_np = []      # 存該 np 下所有 l1 的向量
                all_eigen1_values_np = []   # 存該 np 下所有 l1 的特徵值
                
                # --- 核心修正：對每個 np，遍歷所有 l1 收集特徵 ---
                for l1 in range(L1):
                    # 該 np 下第 l1 段的資料: (B, N1)
                    seg_data = X_k[:, l1*N1:(l1+1)*N1, np_idx] 
                    
                    # 去 DC 算該段的 AC
                    seg_ac = seg_data - (seg_data @ a1_dc_unit) @ a1_dc_unit.T
                    seg_mean = np.mean(seg_ac, axis=0)
                    C_l1_np = ((seg_ac-seg_mean).conj().T @ (seg_ac-seg_mean)) / seg_ac.shape[0]
                    
                    # 取得該段的 N1 個特徵向量與值
                    q_l1_np, eigen_l1_np = self._block_power_method(C_l1_np, N1, num_iter)
                    
                    all_q1_vectors_np.append(q_l1_np)      # (N1, N1)
                    all_eigen1_values_np.append(eigen_l1_np) # 長度 N1
                
                # --- 比較該 np 下的所有 Nt 個 eigenvalues (L1 * N1) ---
                flat_eigen1_np = np.concatenate(all_eigen1_values_np) # 長度 Nt
                global_sort_idx = np.argsort(flat_eigen1_np)[::-1]
                keep_indices = global_sort_idx[:Nt-1] # 去掉最弱的一個，剩 Nt-1
                
                flat_q1_np = np.hstack(all_q1_vectors_np) # (N1, Nt)
                selected_ac_np = flat_q1_np[:, keep_indices] # (N1, Nt-1)
                
                # 補上 DC，組成該 np 的專屬 A1 (N1, Nt)
                A1_np = np.concatenate([a1_dc_unit, selected_ac_np], axis=1)
                self.all_A1_by_np.append(A1_np)
    
            # --- 投影階段：使用每組 np 專屬的 A1 對所有 l1 計算 ---
            # 輸出維度: (B, L1, Nt, Np) -> 總共 L1 * Nt 個 embeddings (對每個 np)
            Z_stage1 = np.zeros((B, L1, Nt, Np), dtype=complex)
            for np_idx in range(Np):
                A1_np = self.all_A1_by_np[np_idx] # (N1, Nt)
                for l1 in range(L1):
                    y_seg = X_k[:, l1*N1:(l1+1)*N1, np_idx] # (B, N1)
                    # 投影產生 (B, Nt) 個係數
                    Z_stage1[:, l1, :, np_idx] = (y_seg @ A1_np.conj()) # 或者是 (A1_np.H @ y.T).T
    
            ########################################
            # Stage 2: Np 維度處理 (比照辦理)
            ########################################
            # 此處輸入為 Z_stage1 (B, L1, Nt, Np)
            # 針對 Np 維度切成長度 N2 的 L2 段
            a2_dc_unit = np.ones((N2, 1)) / np.sqrt(N2)
            Z_stage2 = np.zeros((B, L1, Nt, L2, Np), dtype=complex)
            self.all_A2 = [[None for _ in range(Nt)] for _ in range(L1)]
            for i in range(Nt):
                all_A2_l1 = []
                for l1 in range(L1):
                    # 固定空間特徵，處理 Np 軸
                    z_k_li = Z_stage1[:, l1, i, :] # (B, Np)
                    
                    temp_all_A2 = []
                    all_eigen2_list = []
                    
                    # --- 這裡也要對 l2 進行收集 ---
                    for l2 in range(L2):
                        seg2_data = z_k_li[:, l2*N2:(l2+1)*N2] # (B, N2)
                        seg2_ac = seg2_data - (seg2_data @ a2_dc_unit) @ a2_dc_unit.T
                        seg_mean2 = np.mean(seg2_ac, axis=0)
                        C_l2 = ((seg2_ac-seg_mean2).conj().T @ (seg2_ac-seg_mean2)) / seg2_ac.shape[0]
                        q_l2, eigen_l2 = self._block_power_method(C_l2, N2, num_iter)
                        temp_all_A2.append(q_l2)
                        all_eigen2_list.append(eigen_l2)
                    
                    # 海選 Np 個去掉一個，補 DC
                    flat_eigen2 = np.concatenate(all_eigen2_list)
                    idx2 = np.argsort(flat_eigen2)[::-1]
                    keep2 = idx2[:Np-1]
                    A2 = np.concatenate([a2_dc_unit, np.hstack(temp_all_A2)[:, keep2]], axis=1) # (N2, Np)
    
                    self.all_A2[l1][i] = A2
                    # 對所有 l2 計算投影
                    for l2 in range(L2):
                        z_seg2 = z_k_li[:, l2*N2:(l2+1)*N2]
                        Z_stage2[:, l1, i, l2, :] = (z_seg2 @ A2.conj())
            Z1 = Z_stage2.reshape(B, -1)         
            self.Z1 = self.complex_to_real_imag(Z1)
            return self.Z1
    def module1_operator_DPCA(self, use_module=True, N1=4, N2=4, num_iter=10, client_ranges=None):
            if not use_module:
                self.Z1 = self.X_train.reshape(self.X_train.shape[0], -1)
                return self.Z1
        
            # X shape: (B, 1, Nt, Np)
            B, _, Nt, Np = self.X_train.shape
            X_k = self.X_train[:, 0, :, :] # (B, Nt, Np)
            L1 = Nt // N1
            L2 = Np // N2

            

            ########################################
            # Stage 1: Nt 維度處理 (Spatial)
            ########################################
            a1_dc_unit = np.ones((N1, 1)) / np.sqrt(N1)
            
            # 建立一個容器來存每個 np 的 Anchors
            # 最終我們需要為每個 np 找到 Nt 個 Anchors
            self.all_A1_by_np = [] 
    
            for np_idx in range(Np):
                all_q1_vectors_np = []      # 存該 np 下所有 l1 的向量
                all_eigen1_values_np = []   # 存該 np 下所有 l1 的特徵值
                
                # --- 核心修正：對每個 np，遍歷所有 l1 收集特徵 ---
                for l1 in range(L1):
                    # 該 np 下第 l1 段的資料: (B, N1)
                    seg_data = X_k[:, l1*N1:(l1+1)*N1, np_idx] 
                    
                    # 去 DC 算該段的 AC
                    seg_ac = seg_data - (seg_data @ a1_dc_unit) @ a1_dc_unit.T
                    seg_mean = np.mean(seg_ac, axis=0)
                    C_l1_np = ((seg_ac-seg_mean).conj().T @ (seg_ac-seg_mean)) / seg_ac.shape[0]
                    
                    # 取得該段的 N1 個特徵向量與值
                   
                    q_l1_np, eigen_l1_np = self._block_power_method_fedavg(
                         
                         seg_ac,
                         client_ranges,
                         N1,
                         num_iter
                     )
                    
                    all_q1_vectors_np.append(q_l1_np)      # (N1, N1)
                    all_eigen1_values_np.append(eigen_l1_np) # 長度 N1
                
                # --- 比較該 np 下的所有 Nt 個 eigenvalues (L1 * N1) ---
                flat_eigen1_np = np.concatenate(all_eigen1_values_np) # 長度 Nt
                global_sort_idx = np.argsort(flat_eigen1_np)[::-1]
                keep_indices = global_sort_idx[:Nt-1] # 去掉最弱的一個，剩 Nt-1
                
                flat_q1_np = np.hstack(all_q1_vectors_np) # (N1, Nt)
                selected_ac_np = flat_q1_np[:, keep_indices] # (N1, Nt-1)
                
                # 補上 DC，組成該 np 的專屬 A1 (N1, Nt)
                A1_np = np.concatenate([a1_dc_unit, selected_ac_np], axis=1)
                self.all_A1_by_np.append(A1_np)
    
            # --- 投影階段：使用每組 np 專屬的 A1 對所有 l1 計算 ---
            # 輸出維度: (B, L1, Nt, Np) -> 總共 L1 * Nt 個 embeddings (對每個 np)
            Z_stage1 = np.zeros((B, L1, Nt, Np), dtype=complex)
            for np_idx in range(Np):
                A1_np = self.all_A1_by_np[np_idx] # (N1, Nt)
                for l1 in range(L1):
                    y_seg = X_k[:, l1*N1:(l1+1)*N1, np_idx] # (B, N1)
                    # 投影產生 (B, Nt) 個係數
                    Z_stage1[:, l1, :, np_idx] = (y_seg @ A1_np.conj()) # 或者是 (A1_np.H @ y.T).T
    
            ########################################
            # Stage 2: Np 維度處理 (比照辦理)
            ########################################
            # 此處輸入為 Z_stage1 (B, L1, Nt, Np)
            # 針對 Np 維度切成長度 N2 的 L2 段
            a2_dc_unit = np.ones((N2, 1)) / np.sqrt(N2)
            Z_stage2 = np.zeros((B, L1, Nt, L2, Np), dtype=complex)
            self.all_A2 = [[None for _ in range(Nt)] for _ in range(L1)]
            for i in range(Nt):
                all_A2_l1 = []
                for l1 in range(L1):
                    # 固定空間特徵，處理 Np 軸
                    z_k_li = Z_stage1[:, l1, i, :] # (B, Np)
                    
                    temp_all_A2 = []
                    all_eigen2_list = []
                    
                    # --- 這裡也要對 l2 進行收集 ---
                    for l2 in range(L2):
                        seg2_data = z_k_li[:, l2*N2:(l2+1)*N2] # (B, N2)
                        seg2_ac = seg2_data - (seg2_data @ a2_dc_unit) @ a2_dc_unit.T
                        seg_mean2 = np.mean(seg2_ac, axis=0)
                        C_l2 = ((seg2_ac-seg_mean2).conj().T @ (seg2_ac-seg_mean2)) / seg2_ac.shape[0]
                        q_l2, eigen_l2 = self._block_power_method_fedavg(
                            
                             seg2_ac,
                             client_ranges,
                             N2,
                             num_iter
                         )
                        temp_all_A2.append(q_l2)
                        all_eigen2_list.append(eigen_l2)
                    
                    # 海選 Np 個去掉一個，補 DC
                    flat_eigen2 = np.concatenate(all_eigen2_list)
                    idx2 = np.argsort(flat_eigen2)[::-1]
                    keep2 = idx2[:Np-1]
                    A2 = np.concatenate([a2_dc_unit, np.hstack(temp_all_A2)[:, keep2]], axis=1) # (N2, Np)
    
                    self.all_A2[l1][i] = A2
                    # 對所有 l2 計算投影
                    for l2 in range(L2):
                        z_seg2 = z_k_li[:, l2*N2:(l2+1)*N2]
                        Z_stage2[:, l1, i, l2, :] = (z_seg2 @ A2.conj())
            Z1 = Z_stage2.reshape(B, -1)         
            self.Z1 = self.complex_to_real_imag(Z1)
            return self.Z1
    def module1_operator_HPCA(self, use_module=True, N1=4, N2=4, num_iter=10, client_ranges=None, client_ratios=None):
            if not use_module:
                self.Z1 = self.X_train.reshape(self.X_train.shape[0], -1)
                return self.Z1
        
            # X shape: (B, 1, Nt, Np)
            B, _, Nt, Np = self.X_train.shape
            X_k = self.X_train[:, 0, :, :] # (B, Nt, Np)
            L1 = Nt // N1
            L2 = Np // N2

            

            ########################################
            # Stage 1: Nt 維度處理 (Spatial)
            ########################################
            a1_dc_unit = np.ones((N1, 1)) / np.sqrt(N1)
            
            # 建立一個容器來存每個 np 的 Anchors
            # 最終我們需要為每個 np 找到 Nt 個 Anchors
            self.all_A1_by_np = [] 
    
            for np_idx in range(Np):
                all_q1_vectors_np = []      # 存該 np 下所有 l1 的向量
                all_eigen1_values_np = []   # 存該 np 下所有 l1 的特徵值
                
                # --- 核心修正：對每個 np，遍歷所有 l1 收集特徵 ---
                for l1 in range(L1):
                    # 該 np 下第 l1 段的資料: (B, N1)
                    seg_data = X_k[:, l1*N1:(l1+1)*N1, np_idx] 
                    
                    # 去 DC 算該段的 AC
                    seg_ac = seg_data - (seg_data @ a1_dc_unit) @ a1_dc_unit.T
                    seg_mean = np.mean(seg_ac, axis=0)
                    C_l1_np = ((seg_ac-seg_mean).conj().T @ (seg_ac-seg_mean)) / seg_ac.shape[0]
                    
                    # 取得該段的 N1 個特徵向量與值
                   
                    q_l1_np, eigen_l1_np = self._block_power_method_fedavg_hetero(
                         
                         seg_ac,
                         client_ranges,
                         client_ratios,
                         N1,
                         num_iter
                     )
                    
                    all_q1_vectors_np.append(q_l1_np)      # (N1, N1)
                    all_eigen1_values_np.append(eigen_l1_np) # 長度 N1
                
                # --- 比較該 np 下的所有 Nt 個 eigenvalues (L1 * N1) ---
                flat_eigen1_np = np.concatenate(all_eigen1_values_np) # 長度 Nt
                global_sort_idx = np.argsort(flat_eigen1_np)[::-1]
                keep_indices = global_sort_idx[:Nt-1] # 去掉最弱的一個，剩 Nt-1
                
                flat_q1_np = np.hstack(all_q1_vectors_np) # (N1, Nt)
                selected_ac_np = flat_q1_np[:, keep_indices] # (N1, Nt-1)
                
                # 補上 DC，組成該 np 的專屬 A1 (N1, Nt)
                A1_np = np.concatenate([a1_dc_unit, selected_ac_np], axis=1)
                self.all_A1_by_np.append(A1_np)
    
            # --- 投影階段：使用每組 np 專屬的 A1 對所有 l1 計算 ---
            # 輸出維度: (B, L1, Nt, Np) -> 總共 L1 * Nt 個 embeddings (對每個 np)
            Z_stage1 = np.zeros((B, L1, Nt, Np), dtype=complex)
            for np_idx in range(Np):
                A1_np = self.all_A1_by_np[np_idx] # (N1, Nt)
                for l1 in range(L1):
                    y_seg = X_k[:, l1*N1:(l1+1)*N1, np_idx] # (B, N1)
                    # 投影產生 (B, Nt) 個係數
                    Z_stage1[:, l1, :, np_idx] = (y_seg @ A1_np.conj()) # 或者是 (A1_np.H @ y.T).T
    
            ########################################
            # Stage 2: Np 維度處理 (比照辦理)
            ########################################
            # 此處輸入為 Z_stage1 (B, L1, Nt, Np)
            # 針對 Np 維度切成長度 N2 的 L2 段
            a2_dc_unit = np.ones((N2, 1)) / np.sqrt(N2)
            Z_stage2 = np.zeros((B, L1, Nt, L2, Np), dtype=complex)
            self.all_A2 = [[None for _ in range(Nt)] for _ in range(L1)]
            for i in range(Nt):
                all_A2_l1 = []
                for l1 in range(L1):
                    # 固定空間特徵，處理 Np 軸
                    z_k_li = Z_stage1[:, l1, i, :] # (B, Np)
                    
                    temp_all_A2 = []
                    all_eigen2_list = []
                    
                    # --- 這裡也要對 l2 進行收集 ---
                    for l2 in range(L2):
                        seg2_data = z_k_li[:, l2*N2:(l2+1)*N2] # (B, N2)
                        seg2_ac = seg2_data - (seg2_data @ a2_dc_unit) @ a2_dc_unit.T
                        seg_mean2 = np.mean(seg2_ac, axis=0)
                        C_l2 = ((seg2_ac-seg_mean2).conj().T @ (seg2_ac-seg_mean2)) / seg2_ac.shape[0]
                        q_l2, eigen_l2 = self._block_power_method_fedavg_hetero(
                            
                             seg2_ac,
                             client_ranges,
                             client_ratios,
                             N2,
                             num_iter
                         )
                        temp_all_A2.append(q_l2)
                        all_eigen2_list.append(eigen_l2)
                    
                    # 海選 Np 個去掉一個，補 DC
                    flat_eigen2 = np.concatenate(all_eigen2_list)
                    idx2 = np.argsort(flat_eigen2)[::-1]
                    keep2 = idx2[:Np-1]
                    A2 = np.concatenate([a2_dc_unit, np.hstack(temp_all_A2)[:, keep2]], axis=1) # (N2, Np)
    
                    self.all_A2[l1][i] = A2
                    # 對所有 l2 計算投影
                    for l2 in range(L2):
                        z_seg2 = z_k_li[:, l2*N2:(l2+1)*N2]
                        Z_stage2[:, l1, i, l2, :] = (z_seg2 @ A2.conj())
            Z1 = Z_stage2.reshape(B, -1)         
            self.Z1 = self.complex_to_real_imag(Z1)
            return self.Z1
    ########################################
    # RFT loss
    ########################################
    def _compute_rft_loss(self, feature, B):
        thresholds = np.linspace(np.min(feature), np.max(feature), B+2)[1:-1]
        best_loss = np.inf

        for t in thresholds:
            left = feature < t
            right = ~left

            if np.sum(left) == 0 or np.sum(right) == 0:
                continue

            y_left = self.y_train[left]
            y_right = self.y_train[right]

            loss_left = np.var(y_left)
            loss_right = np.var(y_right)

            loss = (len(y_left)*loss_left + len(y_right)*loss_right) / len(self.y_train)

            if loss < best_loss:
                best_loss = loss

        return best_loss

    ########################################
    # Module 2: Feature selection
    ########################################
    def module2_operator(self, use_module=True, top_k=20):
        if self.Z1 is None:
            raise ValueError("Run module1 first")

        if not use_module:
            self.Z2 = self.Z1
            return self.Z2

        num_features = self.Z1.shape[1]

        scores = []
        for i in range(num_features):
            feature = self.Z1[:, i]  # use real part for splitting
            score = self._compute_rft_loss(feature, B=16)
            scores.append(score)

        scores = np.array(scores)

        idx = np.argsort(scores)[:top_k]
        self.selected_features = idx

        self.Z2 = self.Z1[:, idx]

        return self.Z2
    def module2_operator_sort(self, use_module=True, top_k=20):

        if self.Z1 is None:
            raise ValueError("Run module1 first")

        if not use_module:
            self.Z2 = self.Z1
            self.selected_features = np.arange(self.Z1.shape[1])
            return self.Z2

        num_features = self.Z1.shape[1]
    
        # =====================================================
        # compute RFT scores
        # =====================================================
        scores = np.zeros(num_features)
    
        for i in range(num_features):
            feature = self.Z1[:, i]
            scores[i] = self._compute_rft_loss(feature, B=16)
    
        # =====================================================
        # sort by RFT loss (ascending = better features first)
        # =====================================================
        sorted_idx = np.argsort(scores)
    
        # =====================================================
        # select top-k AND preserve ranking order
        # =====================================================
        selected_idx = sorted_idx[:top_k]
    
        # 🔥 IMPORTANT FIX: enforce ranking order inside Z2
        selected_idx = selected_idx[np.argsort(scores[selected_idx])]
    
        # =====================================================
        # store results
        # =====================================================
        self.selected_features = selected_idx
        self.Z2 = self.Z1[:, selected_idx]
    
        return self.Z2
    ########################################
    # Module 3: MLP (complex → real)
    ########################################
    def module2_operator_plot(self, use_module=True, top_k=20, plot_loss=True):
        if self.Z1 is None:
            raise ValueError("Run module1 first")

        if not use_module:
            self.Z2 = self.Z1
            return self.Z2

        num_features = self.Z1.shape[1]

        scores = []
        for i in range(num_features):
            feature = self.Z1[:, i]  # use real part for splitting
            score = self._compute_rft_loss(feature, B=16)
            scores.append(score)
    
        scores = np.array(scores)
    
        # 排序（由小到大）
        sorted_idx = np.argsort(scores)
        sorted_scores = scores[sorted_idx]
    
        # 取 top_k
        idx = sorted_idx[:top_k]
        self.selected_features = idx
        self.Z2 = self.Z1[:, idx]
    
        # ===== 新增：畫圖 =====
        if plot_loss:
            plt.figure()
            plt.plot(sorted_scores)
            plt.title("Sorted RFT Loss (Ascending)")
            plt.xlabel("Feature Rank")
            plt.ylabel("RFT Loss")
            plt.grid()
            plt.show()
    
        return self.Z2
    def module3_operator(self, epochs=20, batch_size=256):
        if self.Z2 is None:
            raise ValueError("Run module2 first")

        input_dim = self.Z2.shape[1]
        output_dim = self.y_train.shape[1]

        self.model = models.Sequential([
            layers.Input(shape=(input_dim,)),
            
            # 第一層：寬度稍微加大，搭配 BatchNormalization 與複雜一點的 LeakyReLU
            layers.Dense(1024), 
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha=0.1),
            layers.Dropout(0.1), # 輕微的 Dropout 增加泛化能力
            layers.Dense(512),
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha=0.1),
            layers.Dropout(0.2),


            # 輸出層 (保持不變)
            layers.Dense(output_dim)
        ])

        # 使用 AdamW 或調整學習率通常效果更好，這裡維持 adam 
        self.model.compile(optimizer='adam', loss='mse')

        # validation flow (保持不變)
        # 注意：N1, N2 需與訓練時一致，若訓練是 4, 4，這裡也應改為 4, 4
        Z1_val = self._forward_module1(self.X_val, N1=10, N2=3) 
        Z2_val = Z1_val[:, self.selected_features] if self.selected_features is not None else Z1_val
    # 1. 初始化縮放器
        
        train_X_scaled = self.scaler.fit_transform(self.Z2)
        val_X_scaled = self.scaler.transform(Z2_val) # 注意：這裡只用 transform
        self.model.fit(
            train_X_scaled, self.y_train,
            validation_data=(val_X_scaled, self.y_val),
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )
    def module3_operator_xgboost(self,  n_estimators=200, colsample=1, max_depth=6, lr=0.35):
        self.bst = xg.XGBRegressor(n_estimators=n_estimators, 
                              max_depth=max_depth, 
                              gamma=0.1, 
                              tree_method='hist', 
                              gpu_id=0, 
                              colsample_bytree=colsample, 
                              learning_rate=lr, 
                              objective='reg:squarederror',  # 使用回归的目标函数
                              seed=42, reg_lambda = 1000)
        
        self.bst.fit(self.Z2, self.y_train)
    def module3_operator_fedavg(
        self,
        client_ranges,
        global_rounds=20,
        local_epochs=1,
        batch_size=256,
        learning_rate=1e-3,
        N1=10,
        N2=6
    ):
        """
        client_ranges:
            list of tuples
            example:
            [(0,100), (100,200), (200,800), (800,900), (900,1000)]
    
        global_rounds:
            FedAvg communication rounds
    
        local_epochs:
            each client's local training epochs
        """
    
        if self.Z2 is None:
            raise ValueError("Run module2 first")
    
        input_dim = self.Z2.shape[1]
        output_dim = self.y_train.shape[1]
    
        # =========================================================
        # validation flow
        # =========================================================
        Z1_val = self._forward_module1(self.X_val, N1=N1, N2=N2)
    
        if self.selected_features is not None:
            Z2_val = Z1_val[:, self.selected_features]
        else:
            Z2_val = Z1_val
    
        # =========================================================
        # scaler
        # =========================================================
        self.scaler = StandardScaler()
    
        train_X_scaled = self.scaler.fit_transform(self.Z2)
        val_X_scaled = self.scaler.transform(Z2_val)
    
        # =========================================================
        # model builder
        # =========================================================
        def build_model():
    
            model = models.Sequential([
    
                layers.Input(shape=(input_dim,)),
    
                layers.Dense(1024),
                layers.BatchNormalization(),
                layers.LeakyReLU(alpha=0.1),
                layers.Dropout(0.1),
    
                layers.Dense(512),
                layers.BatchNormalization(),
                layers.LeakyReLU(alpha=0.1),
                layers.Dropout(0.2),
    
                layers.Dense(output_dim)
    
            ])
    
            optimizer = tf.keras.optimizers.Adam(
                learning_rate=learning_rate
            )
    
            model.compile(
                optimizer=optimizer,
                loss='mse'
            )
    
            return model
    
        # =========================================================
        # initialize global model
        # =========================================================
        global_model = build_model()
    
        # =========================================================
        # FedAvg rounds
        # =========================================================
        for rnd in range(global_rounds):
    
            print(f"\n========== Global Round {rnd+1}/{global_rounds} ==========")
    
            local_weights = []
            local_num_samples = []
    
            # -----------------------------------------------------
            # each client local training
            # -----------------------------------------------------
            for client_id, (start_idx, end_idx) in enumerate(client_ranges):
    
                # print(f"\nClient {client_id}")
    
                # local dataset
                X_local = train_X_scaled[start_idx:end_idx]
                y_local = self.y_train[start_idx:end_idx]
    
                # build local model
                local_model = build_model()
    
                # load global weights
                local_model.set_weights(
                    global_model.get_weights()
                )
    
                # local update
                local_model.fit(
                    X_local,
                    y_local,
                    epochs=local_epochs,
                    batch_size=batch_size,
                    verbose=0
                )
    
                # store local weights
                local_weights.append(
                    local_model.get_weights()
                )
    
                local_num_samples.append(
                    len(X_local)
                )
    
            # =====================================================
            # FedAvg aggregation
            # =====================================================
            total_samples = np.sum(local_num_samples)
    
            new_global_weights = []
    
            for weights_list_tuple in zip(*local_weights):
    
                weighted_sum = np.zeros_like(weights_list_tuple[0])
    
                for client_idx in range(len(local_weights)):
    
                    client_weight = (
                        local_num_samples[client_idx]
                        / total_samples
                    )
    
                    weighted_sum += (
                        weights_list_tuple[client_idx]
                        * client_weight
                    )
    
                new_global_weights.append(weighted_sum)
    
            # update global model
            global_model.set_weights(new_global_weights)
    
            # =====================================================
            # validation
            # =====================================================
            val_loss = global_model.evaluate(
                val_X_scaled,
                self.y_val,
                verbose=0
            )
    
            print(f"Validation Loss = {val_loss:.6f}")
    
        # =========================================================
        # save final model
        # =========================================================
        self.model = global_model
    
        return global_model
    def module3_operator_fedavg_hetero(
    self,
    client_ranges,
    client_ratios,
    global_rounds=20,
    local_epochs=1,
    batch_size=256,
    learning_rate=1e-3,
    N1=10,
    N2=6
    ):

        if self.Z2 is None:
            raise ValueError("Run module2 first")
    
        input_dim = self.Z2.shape[1]
        output_dim = self.y_train.shape[1]
    
        # =====================================================
        # validation
        # =====================================================
        Z1_val = self._forward_module1(self.X_val, N1=N1, N2=N2)
        Z2_val = Z1_val[:, self.selected_features] if self.selected_features is not None else Z1_val
    
        self.scaler = StandardScaler()
        train_X = self.scaler.fit_transform(self.Z2)
        val_X = self.scaler.transform(Z2_val)
    
        # =====================================================
        # global model (FULL INPUT)
        # =====================================================
        def build_global_model():
            model = models.Sequential([
                layers.Input(shape=(input_dim,)),
                layers.Dense(256),
                layers.LeakyReLU(0.1),
   
    
    
                layers.Dense(output_dim)
            ])
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate),
                loss="mse"
            )
            return model
    
        global_model = build_global_model()
    
        # =====================================================
        # helper: local model builder (variable input dim)
        # =====================================================
        def build_local_model(local_input_dim):
            model = models.Sequential([
                layers.Input(shape=(local_input_dim,)),
                layers.Dense(256),
                layers.LeakyReLU(0.1),
    
                layers.Dense(output_dim)
            ])
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate),
                loss="mse"
            )
            return model
    
        # =====================================================
        # federated learning loop
        # =====================================================
        for rnd in range(global_rounds):
    
            print(f"\n========== Round {rnd+1} ==========")
    
            global_weights = global_model.get_weights()
    
            local_weights_list = []
            local_sample_nums = []
            local_active_dims = []  # 新增：記錄每個 client 實際動用的維度，用於後續精準聚合
    
            # =================================================
            # clients
            # =================================================
            for client_id, (start_idx, end_idx) in enumerate(client_ranges):
    
                ratio = client_ratios[client_id]
                active_dim = max(1, int(input_dim * ratio))
                local_active_dims.append(active_dim)
    
                print(f"Client {client_id} | features = {active_dim}/{input_dim}")
    
                # -------------------------------------------------
                # data (TRUE slicing, NOT masking)
                # -------------------------------------------------
                X_local = train_X[start_idx:end_idx, :active_dim]
                y_local = self.y_train[start_idx:end_idx]
                n_k = X_local.shape[0]
    
                # -------------------------------------------------
                # local model (smaller input space)
                # -------------------------------------------------
                local_model = build_local_model(active_dim)
    
                # -------------------------------------------------
                # initialize from global (projection)
                # -------------------------------------------------
                local_weights = local_model.get_weights()
    
                # 1. 第一層 Kernel 依 active_dim 進行真正裁切 (縱向切片)
                local_weights[0] = global_weights[0][:active_dim, :]
                
                # 2. 自動循環複製其餘所有網路層的權重，避免硬編碼遺漏
                for idx in range(1, len(global_weights)):
                    local_weights[idx] = global_weights[idx].copy()
    
                local_model.set_weights(local_weights)
    
                # -------------------------------------------------
                # local training
                # -------------------------------------------------
                local_model.fit(
                    X_local,
                    y_local,
                    epochs=local_epochs,
                    batch_size=batch_size,
                    verbose=0
                )
    
                # -------------------------------------------------
                # expand back to global dimension
                # -------------------------------------------------
                w_local = local_model.get_weights()
    
                # 創建一個跟 global 第一層相同大小的全零矩陣，再把 local 訓好的部分填回去
                expanded_w0 = np.zeros_like(global_weights[0])
                expanded_w0[:active_dim, :] = w_local[0]
                w_local[0] = expanded_w0
    
                local_weights_list.append(w_local)
                local_sample_nums.append(n_k)
    
            # =====================================================
            # Heterogeneous FedAvg (修正後的聚合機制)
            # =====================================================
            total_samples = np.sum(local_sample_nums)
            new_global_weights = []
    
            for layer_idx in range(len(global_weights)):
                
                if layer_idx == 0:
                    # 【關鍵修改】針對第一層 Heterogeneous Kernel 進行精準分流聚合
                    layer_sum = np.zeros_like(global_weights[0])
                    weight_denominator = np.zeros((input_dim, 1))  # 用來記錄每一行被多少有效樣本訓練過
    
                    for i in range(len(local_weights_list)):
                        n_k = local_sample_nums[i]
                        active_dim_k = local_active_dims[i]
    
                        # 只將 client 實際有訓練到的維度累加進去
                        layer_sum[:active_dim_k, :] += n_k * local_weights_list[i][0][:active_dim_k, :]
                        weight_denominator[:active_dim_k, :] += n_k
    
                    # 進行分母標準化。若某些極高維度在該輪完全沒有 client 覆蓋，則保留原本 global 的值
                    w0_aggregated = np.where(weight_denominator > 0, layer_sum / weight_denominator, global_weights[0])
                    new_global_weights.append(w0_aggregated)
    
                else:
                    # 其餘網路層結構完全一致，走標準的加權平均 (Standard FedAvg)
                    layer_sum = np.zeros_like(global_weights[layer_idx])
                    for i in range(len(local_weights_list)):
                        alpha = local_sample_nums[i] / total_samples
                        layer_sum += alpha * local_weights_list[i][layer_idx]
                    new_global_weights.append(layer_sum)
    
            global_model.set_weights(new_global_weights)
    
            # =====================================================
            # validation
            # =====================================================
            val_loss = global_model.evaluate(val_X, self.y_val, verbose=0)
            print(f"Validation Loss: {val_loss:.6f}")
    
        self.model = global_model
        return global_model
    def NN_operator(self, epochs=20, batch_size=256):
        B, _, Nt, Np = self.X_train.shape
        train_X_complex = self.X_train.reshape(B, -1)         
        train_X = self.complex_to_real_imag(train_X_complex)
        input_dim = train_X.shape[1]
        output_dim = self.y_train.shape[1]

        # 構建 CNN 模型
        inputs = layers.Input(shape=(input_dim,))
    
        # 關鍵步驟：將向量還原為 (Nt, Np, 2) 的圖像格式供 CNN 使用
        # 這裡 2 代表實部與虛部通道
        x = layers.Reshape((Nt, Np, 2))(inputs)
    
        # 第一層卷積：捕捉空間特徵
        x = layers.Conv2D(32, (3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
    
        # 第二層卷積：增加深度
        x = layers.Conv2D(64, (3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
    
        # 第三層卷積：結合殘差概念 (可選)
        shortcut = x
        x = layers.Conv2D(64, (3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Add()([x, shortcut])
        x = layers.Activation('relu')(x)
    
        # 展平回全連接層
        x = layers.Flatten()(x)
        x = layers.Dense(512)(x)
        x = layers.LeakyReLU(alpha=0.1)(x)
        x = layers.Dropout(0.3)(x)
    
        # 輸出層
        outputs = layers.Dense(output_dim)(x)
    
        self.model = Model(inputs=inputs, outputs=outputs)
    
        # 編譯與訓練
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
        self.model.compile(optimizer=optimizer, loss='mse', metrics=['mse'])
    
        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=10, restore_best_weights=True
        )
    # validation flow (保持不變)

        # 注意：N1, N2 需與訓練時一致，若訓練是 4, 4，這裡也應改為 4, 4

        B, _, Nt, Np = self.X_val.shape

        val_X_complex = self.X_val.reshape(B, -1)         

        val_X = self.complex_to_real_imag(val_X_complex)

        train_X_scaled = self.scaler.fit_transform(train_X)
        val_X_scaled = self.scaler.transform(val_X ) # 注意：這裡只用 transform
        print(f"CNN 模型構建完成，輸入維度: {input_dim}, 重組維度: ({Nt}, {Np}, 2)")
        self.model.fit(
            train_X_scaled , self.y_train,
            validation_data=(val_X_scaled, self.y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop],
            verbose=1
        )
    def module3_operator_local_update(self, epochs=20, batch_size=256, learning_rate=1e-4):
        # 1. 防呆檢查：確保模型與 Scaler 已經在第一次訓練時建立
        if not hasattr(self, 'model') or self.model is None:
            raise ValueError("Model not found. 請先執行 module3_operator")
        if not hasattr(self, 'scaler') or self.scaler is None:
            raise ValueError("Scaler not found. 缺少已適配的 Scaler")
    
        # 2. 處理「新」的訓練資料 (讓新的 X_train 通過不動的 Module 1)
        # 這裡的 N1, N2 必須與 predict 和初始訓練時保持一致
        Z1_train = self._forward_module1(self.X_train, N1=10, N2=5)
        self.Z2 = Z1_train[:, self.selected_features] if hasattr(self, 'selected_features') and self.selected_features is not None else Z1_train
        
        # 3. 處理「新」的驗證資料
        Z1_val = self._forward_module1(self.X_val, N1=10, N2=5)
        Z2_val = Z1_val[:, self.selected_features] if hasattr(self, 'selected_features') and self.selected_features is not None else Z1_val
    
        # 4. 資料縮放 (⚠️ 關鍵：嚴格使用 transform，沿用舊的 Scaler 狀態)
        train_X_scaled = self.scaler.transform(self.Z2)
        val_X_scaled = self.scaler.transform(Z2_val)
    
        # 5. 降低學習率並重新編譯 (保留權重，但重置 Adam 的動量以適應新資料)
        from tensorflow.keras.optimizers import Adam
        self.model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')
    
        # 6. 執行增量訓練 (微調)
        print("🚀 開始進行 Module 3 的 Local Update 接續訓練...")
        self.model.fit(
            train_X_scaled, self.y_train,
            validation_data=(val_X_scaled, self.y_val),
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )    
    ########################################
    # Forward helpers
    ########################################
    def _forward_module1(self, X, N1=4, N2=3):
            """
            使用訓練好的 Anchors 對新資料 X 進行投影。
            X shape: (B, 1, Nt, Np) 或 (B, Nt, Np)
            """
            if X.ndim == 4:
                X_k = X[:, 0, :, :]
            else:
                X_k = X
                
            B, Nt, Np = X_k.shape
            L1 = Nt // N1
            L2 = Np // N2
            
            # 準備第一階段輸出容器
            # 根據您的要求，每個 np 會得到 L1 * Nt 個 embeddings
            Z_stage1 = np.zeros((B, L1, Nt, Np), dtype=complex)
            
            ########################################
            # Stage 1 Forward: Spatial Projection
            ########################################
            for np_idx in range(Np):
                # 使用訓練時為該 np 存下的 A1_np (N1, Nt)
                A1_np = self.all_A1_by_np[np_idx] 
                
                for l1 in range(L1):
                    # 取得該段資料 (B, N1)
                    y_seg = X_k[:, l1*N1:(l1+1)*N1, np_idx]
                    
                    # 直接進行投影計算 (B, Nt)
                    # 使用 conj().T 確保複數維度投影正確
                    Z_stage1[:, l1, :, np_idx] = (A1_np.conj().T @ y_seg.T).T
    
            ########################################
            # Stage 2 Forward: Pilot Projection
            ########################################
            # 輸出維度: (B, L1, Nt, L2, Np)
            Z_stage2 = np.zeros((B, L1, Nt, L2, Np), dtype=complex)
            for i in range(Nt):
                for l1 in range(L1):
                    # 拿到 Stage 1 產出的 Np 維度向量 (B, Np)
                    z_k_li = Z_stage1[:, l1, i, :]
                    
                    # 使用訓練時為該 (l1, i) 存下的 A2 (N2, Np)
                    # 這裡假設您在訓練時用了一個字典或 list 儲存：self.all_A2[l1][i]
                    A2_l1_i = self.all_A2[l1][i]
                    
                    for l2 in range(L2):
                        # 取得該段資料 (B, N2)
                        z_seg2 = z_k_li[:, l2*N2:(l2+1)*N2]
                        
                        # 投影產生 (B, Np) 個係數
                        Z_stage2[:, l1, i, l2, :] = (A2_l1_i.conj().T @ z_seg2.T).T
    
            # 最後依照訓練時的邏輯攤平
            # 最終維度應為 (B, L1 * Nt * L2 * Np)
            Z2_complex = Z_stage2.reshape(B, -1)
            Z2 = self.complex_to_real_imag(Z2_complex)

            return Z2
    def predict(self, X, N1=4, N2=3):
    
    
        Z1 = self._forward_module1(X, N1, N2)
    
        Z2 = Z1[:, self.selected_features]

        test_X_scaled = self.scaler.transform(Z2)
        from keras_flops import get_flops
        # Calculae FLOPS
        flops = get_flops(self.model, batch_size=1)
        print(f"FLOPS: {flops / 10 ** 6:.03} M")
        return self.model.predict(test_X_scaled)
    def predict_xgboost(self, X, N1=4, N2=3):
    
    
        Z1 = self._forward_module1(X, N1, N2)
    
        Z2 = Z1[:, self.selected_features]
    
    
        return self.bst.predict(Z2)
    def NN_predict(self, X):
        B, _, Nt, Np = X.shape
        test_X_complex = X.reshape(B, -1)         
        test_X = self.complex_to_real_imag(test_X_complex)

        test_X_scaled = self.scaler.transform(test_X)

    
    
        return self.model.predict(test_X_scaled)
def generate_regression_example(
    num_samples=1000,
    K=2,        # num of AP
    Nt=4,       # antennas per AP
    M=3,        # num of UE
    noise_std=0.01,
    seed=42
):
    np.random.seed(seed)

    # complex Gaussian input
    X_real = np.random.randn(num_samples, K, Nt, M)
    X_imag = np.random.randn(num_samples, K, Nt, M)
    X = X_real + 1j * X_imag

    # coefficients (fixed for determinism)
    alpha = 0.7
    beta = 0.2
    gamma = 0.1

    # regression mapping
    Y = (
        alpha * X
        + beta * np.conj(X)
        + gamma * (np.abs(X) ** 2)
    )

    # add noise
    noise = noise_std * (
        np.random.randn(*X.shape) + 1j * np.random.randn(*X.shape)
    )

    Y = Y + noise

    return X, Y

def flatten_samples_XG(x):
    """
    x: shape = (num_samples, K, M, Nt)
    return: shape = (num_samples * K, M, Nt)
    """
    assert x.ndim == 4, f"Input must be 4D, got {x.ndim}D"
    
    num_samples, K, Nt, M = x.shape
    
    # 先確保記憶體連續（避免 view 出錯）
    x = np.ascontiguousarray(x)
    
    # (num_samples, K, M, Nt) -> (num_samples*K, M, Nt)
    x_flat = x.reshape(num_samples,-1)
    
    return x_flat

def unflatten_samples_AP(x_flat, num_samples, K):
    """
    x_flat: shape = (num_samples * K, 1, M, Nt) 或 (num_samples * K, M, Nt)
    return: shape = (num_samples, K, M, Nt)
    """
    assert x_flat.ndim in [3, 4], f"Input must be 3D or 4D, got {x_flat.ndim}D"

    if x_flat.ndim == 4:
        _, _, M, Nt = x_flat.shape
        x_flat = x_flat.reshape(num_samples, K, 1, M, Nt)
        x = x_flat[:, :, 0, :, :]   # 去掉 channel 維度
    else:
        _, M, Nt = x_flat.shape
        x = x_flat.reshape(num_samples, K, M, Nt)

    return x

def complex_to_real(x):
    return np.concatenate([np.real(x), np.imag(x)], axis=-1)
def load_dataset(file_path):
    data = np.load(file_path)

    X_data_origin = data['X_data_origin']
    Y_label_origin = data['Y_label_origin']
    R_data = data['R_data']
    CSI_origin = data['CSI_origin']

    return X_data_origin, Y_label_origin, R_data, CSI_origin
def flatten_samples_AP(x):
    """
    x: shape = (num_samples, K, M, Nt)
    return: shape = (num_samples * K, M, Nt)
    """
    assert x.ndim == 4, f"Input must be 4D, got {x.ndim}D"
    
    num_samples, K, Nt, M = x.shape
    
    # 先確保記憶體連續（避免 view 出錯）
    x = np.ascontiguousarray(x)
    
    # (num_samples, K, M, Nt) -> (num_samples*K, M, Nt)
    x_flat = x.reshape(num_samples , 1, K*Nt, M)
    
    return x_flat
def augment_with_Pt_SNR(X, Y_origin, CSI, R_data, K, Pt_dBm, SNR_dB):
    """
    使用實際功率 (dBm) + SNR
    ⚠️ 前提：X 已經是 physical scale（不然會不準）
    """

    N = X.shape[0]

    X_rep = np.repeat(X, K, axis=0)
    Y_rep = np.repeat(Y_origin, K, axis=0)
    CSI_rep = np.repeat(CSI, K, axis=0)
    R_rep = np.repeat(R_data, K, axis=0)
    # dBm → Watt
    Pt_watt = 10**((Pt_dBm - 30)/10)

    snr_linear = 10**(SNR_dB / 10)
    noise_power = Pt_watt / snr_linear

    std = np.sqrt(noise_power / 2)

    if np.iscomplexobj(Y_origin):
        noise = std * (
            np.random.randn(*Y_rep.shape) +
            1j * np.random.randn(*Y_rep.shape)
        )
    else:
        noise = std * np.random.randn(*Y_rep.shape)

    Y_aug = Y_rep + noise

    return X_rep, Y_rep, CSI_rep, R_rep
#%%
if __name__ == '__main__':
    # 生成資料
    # X, Y_origin = generate_regression_example(
    # num_samples=1000,
    # K=1,
    # Nt=8,
    # M=4
    # )
    K = 10
    X_data_origin0, Y_label_origin0, R_data1, CSI_origin0 = load_dataset("ue0_100andap0_100_K10_Nt1_M5_Np5_Pt30_noisem90_MMSElabel_imperfectCSI25.npz")
    target_shape = list(X_data_origin0.shape)
    X_data_origin = np.zeros(target_shape, dtype=X_data_origin0.dtype)
    target_shape = list(Y_label_origin0.shape)
    Y_label_origin = np.zeros(target_shape, dtype=Y_label_origin0.dtype)
    target_shape = list(CSI_origin0.shape)
    CSI_origin = np.zeros(target_shape, dtype=CSI_origin0.dtype)
    X_data_origin[:20000] = X_data_origin0[:20000]
    Y_label_origin[:20000] = Y_label_origin0[:20000]
    CSI_origin[:20000] = CSI_origin0[:20000]
    #%%
    X_data_origin0, Y_label_origin0, _, CSI_origin0 = load_dataset("ue0_100andap0_100_K10_Nt1_M4_Np5_Pt30_noisem90_MMSElabel_imperfectCSI25.npz")
    X_data_origin[20000:40000] = X_data_origin0[20000:40000]
    Y_label_origin[20000:40000,:,:,:4] = Y_label_origin0[20000:40000,:,:,:4]
    CSI_origin[20000:40000,:,:4,:] = CSI_origin0[20000:40000,:,:4,:]    
    X_data_origin0, Y_label_origin0, _, CSI_origin0 = load_dataset("ue0_100andap0_100_K10_Nt1_M3_Np5_Pt30_noisem90_MMSElabel_imperfectCSI25.npz")
    X_data_origin[40000:60000] = X_data_origin0[40000:60000]
    Y_label_origin[40000:60000,:,:,:3] = Y_label_origin0[40000:60000,:,:,:3]
    CSI_origin[40000:60000,:,:3,:] = CSI_origin0[40000:60000,:,:3,:]     
    X_data_origin0, Y_label_origin0, _, CSI_origin0 = load_dataset("ue0_100andap0_100_K10_Nt1_M2_Np5_Pt30_noisem90_MMSElabel_imperfectCSI25.npz")
    X_data_origin[60000:80000] = X_data_origin0[60000:80000]
    Y_label_origin[60000:80000,:,:,:2] = Y_label_origin0[60000:80000,:,:,:2]
    CSI_origin[60000:80000,:,:2,:] = CSI_origin0[60000:80000,:,:2,:]      
    X_data_origin0, Y_label_origin0, _, CSI_origin0 = load_dataset("ue0_100andap0_100_K10_Nt1_M1_Np5_Pt30_noisem90_MMSElabel_imperfectCSI25.npz")
    X_data_origin[80000:100000] = X_data_origin0[80000:100000]
    Y_label_origin[80000:100000,:,:,:1] = Y_label_origin0[80000:100000,:,:,:1]
    CSI_origin[80000:100000,:,:1,:] = CSI_origin0[80000:100000,:,:1,:]   
    
    if (K > 1):
        Y_origin_1 = flatten_samples_AP(Y_label_origin)
        X_1 = flatten_samples_AP(X_data_origin)
        CSI_1 = CSI_origin

    else:
        Y_origin_1 = Y_label_origin
        X_1 = X_data_origin
        CSI_1 = CSI_origin

    # 1. 檢查 AP 功率 (目標 23 dBm)
    ap_pows_check = np.sum(np.abs(Y_label_origin)**2, axis=(1, 2, 3)) # (Sample, K)
    max_p_dbm = 10 * np.log10(np.max(ap_pows_check) * 1000)
#%%
    repeat_num = 1
    SNR_dB = 20 
    Ptm = 30
    X, Y_origin, CSI, R_data = augment_with_Pt_SNR(X_1, Y_origin_1, CSI_1, R_data1, repeat_num, Ptm, SNR_dB)
#%%
    # X = flatten_samples_XG(X_origin)
    
    Y = flatten_samples_XG( complex_to_real(Y_origin).astype(np.float32))
    train_ratio=0.7
    val_ratio=0.1
    N = len(X)
    test_size = int(N * (1 - train_ratio - val_ratio))
    X_temp, X_test, Y_temp, Y_test, CSI_temp, CSI_test, R_temp, R_test = train_test_split(
        X, Y, CSI, R_data, test_size= test_size
    )

    val_size = int((N-test_size) *val_ratio / (train_ratio + val_ratio))

    X_train, X_val, Y_train, Y_val, CSI_train, CSI_val, R_train, R_val = train_test_split(
        X_temp, Y_temp, CSI_temp, R_temp, test_size=val_size
    )    


    

#%%    
    # 建立 AP
    ap = AP()
    
    ap.load_dataset_train(X_train, Y_train)
    ap.load_dataset_test(X_test, Y_test)
    ap.load_dataset_val(X_val, Y_val)
#%%
    num_samples = X_train.shape[0]
    num_clients = 5
    
    samples_per_client = num_samples // num_clients
    
    client_ranges = []
    
    for i in range(num_clients):
    
        start_idx = i * samples_per_client
    
        # 最後一個 client 吃剩下全部資料
        if i == num_clients - 1:
            end_idx = num_samples
        else:
            end_idx = (i + 1) * samples_per_client
    
        client_ranges.append((start_idx, end_idx))
    feature_ratios = [1,1,1,1,1]
#%%
    z1_result = ap.module1_operator_HPCA(use_module=True, N1=10, N2=5, num_iter=10, client_ranges=client_ranges, client_ratios=feature_ratios)
    ap.module2_operator(use_module=True, top_k=50)
    ap.module3_operator_fedavg_hetero(client_ranges,
    feature_ratios,                           
    global_rounds=10,
    local_epochs=10,
    batch_size=256,
    learning_rate=1e-3,
    N1=10,
    N2=5)
#%%
    # ap.NN_operator(epochs=10)
    Y_test_predict = ap.predict(X_test, N1=10, N2=5)

    # Y_test_predict = ap.NN_predict(X_test)
#%%
def real_to_complex(x):
    """
    x: (..., 2) 最後一維是 [Re, Im]
    return: (...,) complex
    """
    M = x.shape[-1] // 2
    Re = x[..., :M]
    Im = x[..., M:]
    return Re + 1j * Im


def reconstruct_sum_rate(Y_label, CSI_data, sigma2_dBm=-90, Pt_dBm=50):
    """
    使用 Precoding (W) 與真實通道 (H) 重建 Sum-Rate
    
    參數:
    Y_label: 預編碼矩陣, shape (num_samples, K, Nt, M)
    CSI_data: 真實通道矩陣, shape (num_samples, K, Nt, M)
    sigma2_dBm: 雜訊功率 (預設 -90 dBm)
    
    返回:
    reconstructed_rates: 每個樣本的 Sum-Rate, shape (num_samples,)
    """
    num_samples, K, Nt, M = Y_label.shape
    sigma2 = 10**((sigma2_dBm - 30) / 10)
    reconstructed_rates = np.zeros(num_samples)
    # 1. 定義功率限制
    Pt = 10**((Pt_dBm - 30) / 10) # 轉為線性單位 (瓦特)
    for i in range(num_samples):
        # 1. 重組 H_global: 將各個 AP 的通道拼接到一起
        # CSI_data[i] 是 (K, Nt, M) -> 我們要轉成 (M, K*Nt)
        # 先轉置成 (M, K, Nt) 再 reshape
        H_global = CSI_data[i].transpose(1, 0, 2).reshape(M, K * Nt)
        
        # 2. 重組 W_global: 將各個 AP 的預編碼拼接到一起
        # Y_label[i] 是 (K, Nt, M) -> 轉成 (K*Nt, M)
        W_global = Y_label[i].reshape(K * Nt, M)
        current_power = np.sum(np.abs(W_global)**2)

        # 3. 如果目前功率超過限制，或者為了保持固定功率輸出，進行 Scaling
        if current_power > Pt:
            # 這裡採用等比例縮放，使得 sum(|W|^2) = Pt
            W_global = W_global * np.sqrt(Pt / current_power)
        # ----- Sum-rate -----
        signal = np.zeros(M)
        interf = np.zeros(M, dtype=complex)

        for m in range(M):
            hm = H_global[m, :]           # (N,)
            w_m = W_global[ :, m]           # (N,)
            signal[m] = np.abs(hm @ w_m)**2
        
            for j in range(M):
                if j != m:
                    interf[m] += hm @ W_global[ :, j]

        sinr = signal / (np.abs(interf)**2 + sigma2)
        reconstructed_rates[i] = np.sum(np.log2(1 + sinr))    
    return reconstructed_rates

def unflatten_samples_AP(x_flat, num_samples, K):
    """
    x_flat: shape = (num_samples * K, 1, M, Nt) 或 (num_samples * K, M, Nt)
    return: shape = (num_samples, K, M, Nt)
    """
    assert x_flat.ndim in [3, 4], f"Input must be 3D or 4D, got {x_flat.ndim}D"

    if x_flat.ndim == 4:
        _, _, M, Nt = x_flat.shape
        x_flat = x_flat.reshape(num_samples, K, 1, M, Nt)
        x = x_flat[:, :, 0, :, :]   # 去掉 channel 維度
    else:
        _, M, Nt = x_flat.shape
        x = x_flat.reshape(num_samples, K, M, Nt)

    return x
def unflatten_samples_XG(x_flat, num_samples, K, M, Nt):
    """
    x_flat: shape = (num_samples * K, 1, M, Nt) 或 (num_samples * K, M, Nt)
    return: shape = (num_samples, K, M, Nt)
    """
    # assert x_flat.ndim == 2, f"x_flat must be 4D, got {x_flat.ndim}D"
    x = x_flat.reshape(num_samples, 1, K*Nt, 2*M)

    return x
# --- 驗證重建結果 ---
#NN
# Y_reconstructed_flatten = real_to_complex(Y_pred_nn)
# Y_reconstructed = unflatten_samples_AP(Y_reconstructed_flatten, num_of_samples, K)
#XG
Nt = 1
M = 5
num_te = int(repeat_num*10**5*0.2)
Y_pred_xg = unflatten_samples_XG(Y_test_predict, num_te, K, M, Nt)
Y_reconstructed_flatten = real_to_complex(Y_pred_xg)
# Y_reconstructed = unflatten_samples_AP(Y_reconstructed_flatten, num_te, K)
R_reconstructed = reconstruct_sum_rate(Y_reconstructed_flatten , CSI_test, -90, Ptm)

print(f"原始平均 Sum-Rate: {np.mean(R_test):.4f}")
print(f"重建平均 Sum-Rate: {np.mean(R_reconstructed):.4f}")
# print(f"最大誤差: {np.max(np.abs(R_data - R_reconstructed)):.2e}")