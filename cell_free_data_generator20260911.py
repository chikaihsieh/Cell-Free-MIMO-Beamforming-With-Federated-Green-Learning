# -*- coding: utf-8 -*-
"""
Created on Tue Jan 20 21:47:12 2026

@author: CKAI
"""
import numpy as np

def generate_cell_free_dataset(K = 1, num_samples=100, SNR=20):
    # --- 1. 系統參數設定 ---
    M, Nt = 3, 1
    Np = 2*M
    Pt_dBm, sigma2_dBm = 30, -90
    Pt = 10**((Pt_dBm - 30) / 10)
    sigma2 = 10**((sigma2_dBm - 30) / 10)
    fc, c_light, hAP, hUE = 5.9, 3e8, 25, 1.5
    d_BP = 4 * (hAP - 1 ) * (hUE - 1 ) * fc * 1e9 / c_light
    L = 1
    snr_linear = 10**(SNR / 10)
    # ap_pos = np.array([[0, 0], [-150, 150], [-150, -150], [150, -150]])

    final_X = np.zeros((num_samples, K, Nt, Np), dtype=complex) # 輸入: Received Pilots
    CSI = np.zeros((num_samples, K, M, Nt), dtype=complex) # 輸入: Received Pilots
    final_Y = np.zeros((num_samples, K, Nt, M), dtype=complex) # 標籤: W_k
    distance = np.zeros((num_samples, K, 1, M), dtype=float)
    sum_rate = np.zeros(num_samples)
    init_P = np.random.randn(Np, Np) + 1j*np.random.randn(Np, Np)
    pilot_book, _ = np.linalg.qr(init_P)
    pilot = generate_pilot_fullrank(M, Np, max_retry=10)
    for i in range(num_samples):
        r_min, r_max = 0, 100
        r_ap = np.sqrt(np.random.uniform(r_min**2, r_max**2, K))
        theta_ap0 = np.random.uniform(0, 2 * np.pi, K)
        ap_x = r_ap * np.cos(theta_ap0)
        ap_y = r_ap * np.sin(theta_ap0)
        ap_pos = np.stack((ap_x, ap_y), axis=1)   # shape: (M, 2)
        # ap_pos = np.random.uniform(-100, 100, (K, 2))
        # ue_pos = np.random.uniform(-100, 100, (M, 2))
        theta_ue = np.random.uniform(0, 2 * np.pi, M)
        r_min, r_max = 0, 100
        r_ue = np.sqrt(np.random.uniform(r_min**2, r_max**2, M))
        
        ue_x = r_ue * np.cos(theta_ue)
        ue_y = r_ue * np.sin(theta_ue)
        ue_pos = np.stack((ue_x, ue_y), axis=1)   # shape: (M, 2)

        # ap_pos = np.stack((ap_x, ap_y), axis=1)   # shape: (N, 2)
        ap_angle = np.random.uniform(0, 2*np.pi, K)
        H_global = np.zeros((M, K * Nt), dtype=complex)
        # --- 2. 通道生成 (嚴格遵守投影片 17-20) ---
        for m in range(M):
            for k in range(K):
                d2D = np.linalg.norm(ue_pos[m] - ap_pos[k])
                distance[i, k, :, m] = d2D 
                d3D = np.sqrt(d2D**2 + (hAP - hUE)**2)
                
                # Path Loss (頁面 19-20)
                if d2D <= d_BP:
                    pl_los = 28.0 + 22 * np.log10(d3D) + 20 * np.log10(fc)
                else:
                    pl_los = 28.0 + 40 * np.log10(d3D) + 20 * np.log10(fc) - 9 * np.log10(d_BP**2 + (hAP-hUE)**2)
                
                # pr_los = 1.0 if d2D <= 18 else (18/d2D + np.exp(-d2D/63)*(1 - 18/d2D))
                final_pl = pl_los
                # final_pl = max(pl_los, 13.54 + 39.08 * np.log10(d3D) + 20 * np.log10(fc) - 0.6 * (hUE - 1.5))
                kappa = 4
                # Beta 修正：10^-PL/20 且不再 sqrt
                beta = 10**(-final_pl / 10)
                
                # 小尺度 3-path (頁面 17)
                # 1. 計算 UE 相對於 AP 的方位角 (幾何角度)
                # ue_pos[m] 是 (x, y), ap_pos[k] 是 (x, y)
                dx = ue_pos[m, 0] - ap_pos[k, 0]
                dy = ue_pos[m, 1] - ap_pos[k, 1]
                theta_geo = np.arctan2(dy, dx) - ap_angle[k]
                # 2. 替換原本的隨機角度
                # 將 np.random.uniform(0, np.pi) 換成 theta_geo
                h_los = np.exp(1j * np.pi * np.arange(Nt) *np.sin(theta_geo))
                h_nlos = np.zeros(Nt, dtype=complex)
                for _ in range(L):
                    h_nlos += ((np.random.randn() + 1j*np.random.randn()) / np.sqrt(2*L)) *np.exp(1j * np.pi * np.arange(Nt) * np.sin(np.random.uniform(0, 2*np.pi)))
                H_global[m, k*Nt : (k+1)*Nt] = np.sqrt(beta) * (np.sqrt(kappa/(kappa+1))*h_los + np.sqrt(1/(kappa+1))*h_nlos)

        # --- 3. ZF 與 Water-filling (頁面 23-24) ---
        W_tilde = np.zeros(( K*Nt , M), dtype=complex)
        W_final = np.zeros(( K*Nt , M), dtype=complex)
        CSI_power = np.sum(np.abs(H_global)**2)
        imperfect_noise_power = CSI_power / snr_linear
        imperfect_std = np.sqrt(imperfect_noise_power / 2)
        imperfect_noise = imperfect_std * (
            np.random.randn(*H_global.shape) +
            1j * np.random.randn(*H_global.shape)
        )
        H = H_global + imperfect_noise   
        temp = np.linalg.inv( H @ H.conj().T+ np.eye(M)*M*(sigma2/Pt))
        temp2 = H.conj().T 
        temp3 = temp2 @temp
        W_tilde = temp3
        wm = np.real(np.diag(  W_tilde.conj().T @ W_tilde))
        l_low, l_high = 1e-12, 1e12
        for _ in range(500):
            lam = 0.5 * (l_low + l_high)
            p = 1/wm*np.maximum(1/lam - sigma2 * wm, 0)
            if np.sum(p*wm) > Pt:
                l_low = lam
            else:
                l_high = lam

        W_final =  W_tilde @ np.sqrt(np.diag(p))
            
            
        # --- 4. 存入 X (接收信號 Y) 與 Y (標籤 W) ---
        for k in range(K):
            H_k = H_global[:, k*Nt : (k+1)*Nt]
            # 模擬收到 Pilot 信號 (頁面 9): Y = H + N
            noise = (np.random.randn(Nt, Np) + 1j*np.random.randn(Nt, Np)) * np.sqrt(sigma2/2)
            # Received pilot
            final_X[i, k, :, :] = (H_k.conj().T @ pilot + noise) 
            CSI[i, k, :, :] = H_k 
            final_Y[i, k, :, :] = W_final[ k*Nt : (k+1)*Nt, :]
            # Ptemp = np.sum(np.abs(W_final[k*Nt : (k+1)*Nt, :])**2)
            # if( Ptemp > Pt):
            #     final_Y[i, k, :, :] = W_final[k*Nt : (k+1)*Nt, :]*Pt/Ptemp
            #     W_final[k*Nt : (k+1)*Nt, :] = W_final[k*Nt : (k+1)*Nt, :]*Pt/Ptemp
        # ----- Sum-rate -----
        signal = np.zeros(M)
        interf = np.zeros(M, dtype=complex)

        for m in range(M):
            hm = H_global[m, :]           # (N,)
            w_m = W_final[ :, m]           # (N,)
            signal[m] = np.abs(hm @ w_m)**2
        
            for j in range(M):
                if j != m:
                    interf[m] += hm @ W_final[ :, j]

        sinr = signal / (np.abs(interf)**2 + sigma2)
        sum_rate[i] = np.sum(np.log2(1 + sinr))
    return final_X, final_Y, sum_rate, CSI, distance

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
    x_flat = x.reshape(num_samples * K, 1, Nt, M)
    
    return x_flat

def generate_pilot_fullrank(M, Np, max_retry=10):
    for _ in range(max_retry):
        pilot = np.random.randn(M, Np) + 1j * np.random.randn(M, Np)
        pilot = pilot / np.linalg.norm(pilot, axis=1, keepdims=True)

        if np.linalg.matrix_rank(pilot) == M:
            return pilot

    raise ValueError("Failed to generate full-rank pilot matrix")


# --- 執行與檢查指令 ---
K = 10
num_of_samples = 100000
SNR = 20
X_data_origin, Y_label_origin, R_data, CSI_origin, distance_origin = generate_cell_free_dataset(K, num_of_samples, SNR)

if (K > 1):
    Y_label = flatten_samples_AP(Y_label_origin)
    X_data = flatten_samples_AP(X_data_origin)
    CSI = flatten_samples_AP(CSI_origin)
    dist = flatten_samples_AP(distance_origin)
else:
    Y_label = Y_label_origin
    X_data = X_data_origin
    CSI = CSI_origin
    dist = distance_origin
# 1. 檢查 AP 功率 (目標 23 dBm)
ap_pows_check = np.sum(np.abs(Y_label_origin)**2, axis=(1, 2, 3)) # (Sample, K)
max_p_dbm = 10 * np.log10(np.max(ap_pows_check) * 1000)
print(f"最大 AP 功率: {max_p_dbm:.2f} dBm (應接近 )")

# 2. 檢查 Rate
print(f"平均 Sum-Rate: {np.mean(R_data):.2f} bps/Hz")
# 3. 檢查維度 (用於 GL 模型)
print(f"輸入 X 維度: {X_data.shape}") # (Sample, AP, Nt, M)
print(f"標籤 Y 維度: {Y_label.shape}") # (Sample, AP, Nt, M)

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


def reconstruct_sum_rate(Y_label, CSI_data, sigma2_dBm=-90):
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
    
    for i in range(num_samples):
        # 1. 重組 H_global: 將各個 AP 的通道拼接到一起
        # CSI_data[i] 是 (K, Nt, M) -> 我們要轉成 (M, K*Nt)
        # 先轉置成 (M, K, Nt) 再 reshape
        H_global = CSI_data[i].transpose(1, 0, 2).reshape(M, K * Nt)
        
        # 2. 重組 W_global: 將各個 AP 的預編碼拼接到一起
        # Y_label[i] 是 (K, Nt, M) -> 轉成 (K*Nt, M)
        W_global = Y_label[i].reshape(K * Nt, M)
        
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
# --- 驗證重建結果 ---
#NN
# Y_reconstructed_flatten = real_to_complex(Y_pred_nn)
# Y_reconstructed = unflatten_samples_AP(Y_reconstructed_flatten, num_of_samples, K)
#XG
# Y_reconstructed_flatten = real_to_complex(Y_pred_xg)
# Y_reconstructed = unflatten_samples_AP(Y_reconstructed_flatten, num_of_samples, K)
R_reconstructed = reconstruct_sum_rate(Y_label_origin, CSI_origin)

print(f"原始平均 Sum-Rate: {np.mean(R_data):.4f}")
print(f"重建平均 Sum-Rate: {np.mean(R_reconstructed):.4f}")
# print(f"最大誤差: {np.max(np.abs(R_data - R_reconstructed)):.2e}")
#%%
def save_dataset(file_path, X_data_origin, Y_label_origin, R_data, CSI_origin):
    """
    X_data_origin, Y_label_origin, R_data: shape = (num_of_samples, a, b, c)
    CSI_origin: shape = (num_of_samples, 1)
    """
    np.savez(file_path,
             X_data_origin=X_data_origin,
             Y_label_origin=Y_label_origin,
             R_data=R_data,
             CSI_origin=CSI_origin)
    
save_dataset("K1_Nt4_M3_Np6_Pt30_noisem90_MMSElabel.npz",
             X_data_origin,
             Y_label_origin,
             R_data,
             CSI_origin)