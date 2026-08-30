# -*- coding: utf-8 -*-
"""生成图22：训练外工况相位对齐总波场响应的真值、预测与残差场。"""

import pickle
from pathlib import Path

import matplotlib
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np


matplotlib.use("Agg")

REPO_ROOT = Path(__file__).resolve().parents[5]
DATA_PATH = REPO_ROOT / "Run" / "ch4_sp_analysis" / "complex_frf_dataset.npz"
MODEL_PATH = REPO_ROOT / "Run" / "ch4_sp_ml" / "complex_frf_surrogate.pkl"
OUTPUT_DIR = Path(__file__).resolve().parent
OUTPUT_STEM = "fig22_训练外真值预测残差场"
CASE_ORDER = ("B002", "B003", "B004")
CASE_LEVEL = {"B002": "好", "B003": "中", "B004": "难"}


def set_journal_style():
    """设置中文期刊绘图样式。"""
    for font_path in (
        r"C:\Windows\Fonts\simsun.ttc",
        r"C:\Windows\Fonts\STSONG.TTF",
        r"C:\Windows\Fonts\simhei.ttf",
    ):
        try:
            fm.fontManager.addfont(font_path)
        except Exception:
            pass
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["SimSun", "STSong", "SimHei", "DejaVu Sans"],
            "mathtext.fontset": "stix",
            "axes.unicode_minus": False,
            "axes.linewidth": 0.8,
            "axes.labelsize": 9,
            "axes.titlesize": 9.2,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def decode_strings(values):
    """将NPZ中的字节串或字符串统一转为普通字符串。"""
    return [item.decode("utf-8") if isinstance(item, bytes) else str(item) for item in values]


def predict_field(model, X):
    """按冻结模型字典预测完整相位对齐总波场响应。"""
    X = np.atleast_2d(np.asarray(X, dtype=float))
    X_scaled = model["feature_scaler"].transform(X)
    if model["model_name"] == "nearest":
        distances = np.sum((X_scaled[:, None, :] - model["X_scaled"][None, :, :]) ** 2, axis=2)
        indices = np.argmin(distances, axis=1)
        return model["field"][indices].copy(), model["valid_mask"][indices].copy()
    estimator = model["estimator"]
    if model["model_name"] == "pod_gpr":
        scores = np.column_stack([item.predict(X_scaled) for item in estimator])
    else:
        scores = np.asarray(estimator.predict(X_scaled))
        if scores.ndim == 1:
            scores = scores[:, None]
    standardized = scores.dot(model["basis"])
    values = standardized * model["scale"] + model["center"]
    pixel_count = int(np.sum(model["pixel_mask"]))
    complex_values = values[:, :pixel_count] + 1j * values[:, pixel_count:]
    output = np.full((len(X),) + tuple(model["field_shape"]), np.nan + 1j * np.nan, complex)
    output[:, model["pixel_mask"]] = complex_values
    mask = np.broadcast_to(model["pixel_mask"], output.shape).copy()
    return output, mask


def weighted_complex_error(truth, prediction, valid, frequency_weight):
    """计算与代理评价一致的频率加权复数相对误差。"""
    weights = np.broadcast_to(np.asarray(frequency_weight, float)[:, None], truth.shape)
    use = valid & np.isfinite(truth) & np.isfinite(prediction)
    numerator = np.sum(weights[use] * np.abs(prediction[use] - truth[use]) ** 2)
    denominator = np.sum(weights[use] * np.abs(truth[use]) ** 2)
    return float(np.sqrt(numerator / denominator))


def save_figure(fig):
    """输出300 dpi PNG。"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for suffix in (".png",):
        path = OUTPUT_DIR / (OUTPUT_STEM + suffix)
        fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
        print("已生成：%s" % path)


def main():
    """读取统一左参考数据集和冻结代理包并绘制三类训练外工况。"""
    set_journal_style()
    if not DATA_PATH.exists() or not MODEL_PATH.exists():
        raise FileNotFoundError("缺少总波场响应数据集或冻结代理包。")

    dataset = np.load(DATA_PATH, allow_pickle=True)
    case_ids = decode_strings(dataset["case_ids"])
    frequency = np.asarray(dataset["frequency_hz"], float)
    s_values = np.asarray(dataset["s"], float)
    feature_names = decode_strings(dataset["feature_names"])
    with MODEL_PATH.open("rb") as handle:
        bundle = pickle.load(handle)
    model = bundle["G_h_model"]

    records = []
    for case_id in CASE_ORDER:
        index = case_ids.index(case_id)
        truth = np.asarray(dataset["G_h"][index], complex)
        truth_mask = np.asarray(dataset["valid_mask"][index], bool)
        parameters = np.asarray(dataset["X"][index], float)
        prediction, prediction_mask = predict_field(model, parameters)
        prediction = prediction[0]
        valid = truth_mask & prediction_mask[0]
        error = weighted_complex_error(truth, prediction, valid, dataset["input_weight"][index])
        amplitude_truth = np.where(valid, np.abs(truth), np.nan)
        amplitude_prediction = np.where(valid, np.abs(prediction), np.nan)
        log_residual = np.where(
            valid,
            np.log(np.maximum(np.abs(prediction), 1.0e-12) / np.maximum(np.abs(truth), 1.0e-12)),
            np.nan,
        )
        phase_residual = np.where(valid, np.degrees(np.angle(prediction * np.conj(truth))), np.nan)
        records.append(
            {
                "case_id": case_id,
                "parameters": parameters,
                "truth": amplitude_truth,
                "prediction": amplitude_prediction,
                "log_residual": log_residual,
                "phase_residual": phase_residual,
                "error": error,
            }
        )

    amplitudes = np.concatenate(
        [item[key][np.isfinite(item[key])] for item in records for key in ("truth", "prediction")]
    )
    log_errors = np.concatenate([item["log_residual"][np.isfinite(item["log_residual"])] for item in records])
    phase_errors = np.concatenate(
        [item["phase_residual"][np.isfinite(item["phase_residual"])] for item in records]
    )
    amp_max = float(np.percentile(amplitudes, 99.5))
    log_limit = max(0.25, float(np.percentile(np.abs(log_errors), 98.5)))
    phase_limit = max(15.0, float(np.percentile(np.abs(phase_errors), 98.5)))

    amplitude_cmap = plt.get_cmap("cividis").copy()
    residual_cmap = plt.get_cmap("RdBu_r").copy()
    amplitude_cmap.set_bad("#EEEEEE")
    residual_cmap.set_bad("#EEEEEE")

    fig, axes = plt.subplots(3, 4, figsize=(13.2, 9.1), sharex=True, sharey=True, constrained_layout=True)
    extent = (s_values[0], s_values[-1], frequency[0], frequency[-1])
    images = [None, None, None, None]
    for row, item in enumerate(records):
        fields = (item["truth"], item["prediction"], item["log_residual"], item["phase_residual"])
        specs = (
            (amplitude_cmap, 0.0, amp_max),
            (amplitude_cmap, 0.0, amp_max),
            (residual_cmap, -log_limit, log_limit),
            (residual_cmap, -phase_limit, phase_limit),
        )
        for column, (field, spec) in enumerate(zip(fields, specs)):
            cmap, vmin, vmax = spec
            images[column] = axes[row, column].imshow(
                field,
                origin="lower",
                aspect="auto",
                extent=extent,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                interpolation="nearest",
                rasterized=True,
            )
            axes[row, column].axvline(0.0, color="white", lw=0.65, ls="--", alpha=0.9)
            axes[row, column].axvline(1.0, color="white", lw=0.65, ls="--", alpha=0.9)
            axes[row, column].set_xlim(-4.0, 4.0)
            axes[row, column].set_ylim(0.5, 10.0)
        i_value, d_value, rv_value = item["parameters"]
        axes[row, 0].set_ylabel("频率 $f$/Hz")
        axes[row, 0].text(
            -0.33,
            0.50,
            "%s（%s）\n$i$=%.1f°\n$d/h$=%.2f\n$r_v$=%.3f\n$E_c$=%.3f"
            % (item["case_id"], CASE_LEVEL[item["case_id"]], i_value, d_value, rv_value, item["error"]),
            transform=axes[row, 0].transAxes,
            ha="center",
            va="center",
            fontsize=8.2,
            linespacing=1.35,
        )
        axes[row, 0].text(-3.70, 9.55, "A", color="white", weight="bold", fontsize=8.5)
        axes[row, 0].text(0.45, 9.55, "B", color="white", weight="bold", fontsize=8.5)
        axes[row, 0].text(2.35, 9.55, "C", color="white", weight="bold", fontsize=8.5)

    titles = ("真值 $|G_h|$", "POD-GPR预测 $|G_h|$", "对数幅值残差", "圆周相位残差")
    for column, title in enumerate(titles):
        axes[0, column].set_title("(%s) %s" % (chr(ord("a") + column), title), loc="left")
        axes[-1, column].set_xlabel("归一化地表坐标 $s$")

    cbar_amp = fig.colorbar(images[0], ax=axes[:, :2].ravel().tolist(), orientation="horizontal", shrink=0.52, pad=0.035)
    cbar_amp.set_label("相位对齐总波场响应幅值 $|G_h|$")
    cbar_log = fig.colorbar(images[2], ax=axes[:, 2].ravel().tolist(), orientation="horizontal", shrink=0.78, pad=0.035)
    cbar_log.set_label(r"$\ln(|\widehat{G}_h|/|G_h|)$")
    cbar_phase = fig.colorbar(images[3], ax=axes[:, 3].ravel().tolist(), orientation="horizontal", shrink=0.78, pad=0.035)
    cbar_phase.set_label(r"$\angle(\widehat{G}_hG_h^*)$/(°)")
    save_figure(fig)
    plt.close(fig)

    print("数据来源：%s；%s" % (DATA_PATH, MODEL_PATH))
    for item in records:
        log_med = float(np.nanmedian(np.abs(item["log_residual"])))
        phase_med = float(np.nanmedian(np.abs(item["phase_residual"])))
        print(
            "%s：E_complex,w=%.3f，|对数幅值残差|中位=%.3f，|相位残差|中位=%.1f°"
            % (item["case_id"], item["error"], log_med, phase_med)
        )


if __name__ == "__main__":
    main()
