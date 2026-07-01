# -*- coding: utf-8 -*-
"""EQL(等效线性)一维场地反应原型：可切换经验曲线 + SHAKE 式迭代 + 强度扫描。
纯 numpy，不依赖 Abaqus。目的：把物理跑通、看放大随强度的变化，再接入 fd 引擎。"""
import numpy as np

# ============ 一、可切换经验曲线：G/Gmax 与 阻尼比 ξ 随剪应变 γ ============
def mod_damp(gamma, curve='darendeli', PI=0.0, sigma0_kpa=100.0):
    """返回 (G/Gmax, xi)。gamma=工程剪应变(小数)。三种曲线可切换，便于对比。
    统一用双曲骨架 G/Gmax=1/(1+(γ/γref)^a)，γref/a 与最小/上限阻尼按模型取值（工程近似）。"""
    g = np.maximum(np.abs(gamma), 1e-9)
    if curve == 'seed_idriss_sand':          # 砂/砾(Seed-Idriss 均值)
        gref, a = 6.0e-4, 0.92               # 参考应变~0.06%
        xi_min, xi_max, k = 0.010, 0.28, 0.33
    elif curve == 'vucetic_dobry':           # 黏土(随塑性指数 PI 变线性)
        gref, a = 3.0e-4*(1.0+PI/30.0), 0.85
        xi_min, xi_max, k = max(0.008, 0.025-0.0001*PI), 0.25, 0.30
    else:                                    # darendeli(2001 通用，含围压/PI)
        gref = (0.0352 + 0.0010*PI) * (sigma0_kpa/100.0)**0.35 / 100.0  # →小数
        a = 0.919
        xi_min, xi_max, k = 0.008 + 0.0005*PI/10.0, 0.24, 0.31
    GG = 1.0/(1.0+(g/gref)**a)               # 模量折减(双曲骨架)
    xi = np.clip(xi_min + k*(1.0-GG), xi_min, xi_max)  # 阻尼随软化增大(工程近似)
    return GG, xi

# ============ 二、分层 SH：地表放大 与 各层中点应变 传递函数 ============
def sh_transfer(freqs, Vs, rho, h, xi, Vb, rhob, xib, fcut=15.0):
    """竖直入射 SH，传播矩阵法。返回 (放大谱 surf/outcrop, 各层中点应变谱 strain/outcrop_disp)。
    Vs,rho,h,xi: 各有限层(从上到下)数组；Vb,rhob,xib: 基岩半空间。"""
    nL = len(Vs)
    amp = np.zeros(len(freqs), complex)
    strain_mid = np.zeros((nL, len(freqs)), complex)  # 各层中点应变TF
    def prop(kh, mu, k):              # 解析传播逆(避免对大幅值矩阵做数值求解): bottom=Pinv·top
        c, sn = np.cos(kh), np.sin(kh)   # det(P)=1, 逆=[[c,-sn/(μk)],[μk·sn,c]]
        return c, sn
    for fi, f in enumerate(freqs):
        w = 2*np.pi*f
        if w == 0 or f > fcut:            # 跳过零频与高频(输入无能量,避免溢出)
            amp[fi]=1.0; continue
        try:
            s = np.array([1.0+0j, 0.0+0j])    # 地表 [u=1, τ=0]
            for j in range(nL):               # 顶层在前
                b = Vs[j]*np.sqrt(1+2j*xi[j]); mu = rho[j]*b*b; k = w/b
                c,sn = np.cos(k*h[j]/2), np.sin(k*h[j]/2)     # 传到层中点
                u_mid = c*s[0] - sn/(mu*k)*s[1]               # 逆矩阵解析: u
                t_mid = mu*k*sn*s[0] + c*s[1]                 # 逆矩阵解析: τ
                strain_mid[j, fi] = t_mid/mu                 # γ_mid=τ/μ
                c,sn = np.cos(k*h[j]), np.sin(k*h[j])         # 传过整层到层底
                s = np.array([c*s[0]-sn/(mu*k)*s[1], mu*k*sn*s[0]+c*s[1]])
            uH, tH = s
            bb = Vb*np.sqrt(1+2j*xib); mub = rhob*bb*bb; kb = w/bb
            A = (uH + 1j*tH/(kb*mub))/2.0
            if not np.isfinite(A) or abs(A)<1e-30:
                amp[fi]=0.0; strain_mid[:,fi]=0.0; continue
            amp[fi] = 1.0/(2*A); strain_mid[:, fi] /= (2*A)
        except Exception:
            amp[fi]=0.0; strain_mid[:,fi]=0.0
    return amp, strain_mid

# ============ 三、EQL 迭代(SHAKE) ============
def run_eql(acc_in, dt, Vs0, rho, h, Vb, rhob, nonlin, curve='darendeli', PI=0.0,
            sigma0_kpa=100.0, ratio=0.65, tol=0.02, maxit=15):
    """对输入 outcrop 加速度 acc_in 做 EQL，返回各层收敛 Vs/GG/ξ/γ_eff 与地表放大谱。
    nonlin: 各层是否非线性(布尔数组)；线性层固定 ξ=0.02。"""
    n = len(acc_in)
    A = np.fft.rfft(acc_in); fr = np.fft.rfftfreq(n, dt); w = 2*np.pi*fr
    Udisp = np.zeros_like(A); Udisp[1:] = -A[1:]/w[1:]**2   # outcrop 位移谱
    Vs = np.array(Vs0, float); xi = np.where(nonlin, 0.02, 0.02)
    GG = np.ones(len(Vs)); geff = np.zeros(len(Vs)); xib = 0.005
    for it in range(maxit):
        amp, strainTF = sh_transfer(fr, Vs, rho, h, xi, Vb, rhob, xib)
        Vs_new = Vs.copy(); xi_new = xi.copy()
        for j in range(len(Vs)):
            if not nonlin[j]:
                continue
            gt = np.fft.irfft(strainTF[j]*Udisp, n=n)   # γ(t)
            geff[j] = ratio*np.max(np.abs(gt))          # 有效应变
            gg, x = mod_damp(geff[j], curve, PI, sigma0_kpa)
            GG[j] = gg
            Vs_new[j] = Vs0[j]*np.sqrt(gg)              # Vs=Vs0·√(G/Gmax)
            xi_new[j] = x
        rel = np.max(np.abs(Vs_new-Vs)/np.maximum(Vs,1e-9))
        Vs, xi = Vs_new, xi_new
        if rel < tol:
            break
    amp, _ = sh_transfer(fr, Vs, rho, h, xi, Vb, rhob, xib)
    return {'Vs':Vs,'GG':GG,'xi':xi,'geff':geff,'amp':amp,'fr':fr,'iters':it+1}

# ============ 四、演示：软厚层柱，按强度扫描，三曲线对比 ============
if __name__ == '__main__':
    # 软厚层柱(Fig.15b)：表层400/150m，覆盖800/250m，基岩2000
    Vs0=np.array([400.,800.]); rho=np.array([2500.,2500.]); h=np.array([150.,250.])
    Vb,rhob=2000.,2500.; nonlin=np.array([True,False])   # 仅软表层非线性
    dt=0.001; t=np.arange(0,4,dt); fc=2.0; t0=0.5        # 2Hz Ricker(打中软层共振)
    arg=(np.pi**2)*(fc**2)*(t-t0)**2; ric=(1-2*arg)*np.exp(-arg)
    print('柱: 软表层 Vs=400/150m(非线性) + 覆盖 Vs=800/250m(线性) + 基岩2000')
    for curve in ['seed_idriss_sand','vucetic_dobry','darendeli']:
        print('\n=== 曲线: %s ==='%curve)
        print('%6s %9s %9s %8s %10s'%('PGA(g)','γ_eff(%)','G/Gmax','ξ(%)','峰值放大'))
        for pga_g in [0.02,0.05,0.1,0.2,0.35,0.5]:
            acc = ric/np.max(np.abs(ric))*(pga_g*9.81)   # 缩放到目标 outcrop PGA
            r = run_eql(acc, dt, Vs0, rho, h, Vb, rhob, nonlin, curve=curve, PI=15.0)
            print('%6.2f %9.3f %9.3f %8.1f %10.2f'%(
                pga_g, r['geff'][0]*100, r['GG'][0], r['xi'][0]*100, np.max(np.abs(r['amp']))))
