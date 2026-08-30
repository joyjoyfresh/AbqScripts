# -*- coding: utf-8 -*-
"""批量导出各图的CSV源数据表，供论文投稿提交图源数据使用。

以子进程方式逐个运行figures下各图文件夹中的出图脚本，运行前拦截matplotlib
的Axes绘图方法以捕获数值数据：曲线、散点、柱状、箱线、参考线等统一写入
figXX_图名.csv（长表，列为“子图,系列,x,y”）；热图子图（pcolormesh/imshow）
另存figXX_图名_热图N.csv（矩阵表，行y列x）。直接运行处理全部图；传入单个
图脚本路径时只处理该图。出图脚本自身不需要任何修改。
"""

from __future__ import annotations

import csv
import os
import runpy
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
from matplotlib.axes import Axes


FIGURES_ROOT = Path(__file__).resolve().parent
RECORDS: list[dict] = []  # 按创建顺序记录每个Axes及其绘图操作


def clean_label(label) -> str:
    """把图例标签中的LaTeX记号转成CSV可读的纯文本。"""
    if label is None:
        return ""
    s = str(label).strip()
    if not s or s.startswith("_"):
        return ""
    for old, new in (
        ("$", ""),
        ("\\circ", "°"),
        ("\\geq", "≥"),
        ("\\leq", "≤"),
        ("\\times", "×"),
        ("\\sim", "~"),
        ("\\_", "_"),
        ("\\", ""),
        ("{", ""),
        ("}", ""),
        ("^", ""),
    ):
        s = s.replace(old, new)
    return s.strip()


def fmt(value) -> str:
    """数值转字符串，非有限值输出为空。"""
    try:
        array = np.asarray(value)
        if array.ndim != 0:
            return ""
        number = array.item()
    except Exception:
        return ""
    if number is None or not np.isfinite(number):
        return ""
    return "%.10g" % number


def array_copy(value):
    """把输入安全转为浮点数组。"""
    try:
        array = np.asarray(value, dtype=float)
    except Exception:
        return None
    if array.ndim == 0:
        return array.reshape(1)
    return array


def axis_record(axis) -> dict:
    """获取或创建一个Axes的捕获记录。"""
    for record in RECORDS:
        if record["axis"] is axis:
            return record
    record = {"axis": axis, "ops": []}
    RECORDS.append(record)
    return record


def wrap_axes_method(name, handler):
    """拦截一个Axes方法，在原方法执行后保存数据。"""
    original = getattr(Axes, name)

    def wrapped(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        try:
            handler(axis_record(self), args, kwargs, result)
        except Exception as exc:
            axis_record(self).setdefault("warnings", []).append("%s: %s" % (name, exc))
        return result

    setattr(Axes, name, wrapped)


def first_arg(args, kwargs, *names):
    """取位置参数或关键字参数。"""
    if args:
        return args[0]
    for name in names:
        if name in kwargs:
            return kwargs[name]
    return 0.0


def handle_plot(record, args, kwargs, result):
    for line in result:
        record["ops"].append(
            {
                "kind": "line",
                "x": array_copy(line.get_xdata(orig=False)),
                "y": array_copy(line.get_ydata(orig=False)),
                "label": clean_label(kwargs.get("label", "")) or "",
            }
        )


def handle_scatter(record, args, kwargs, result):
    offsets = np.asarray(result.get_offsets(), dtype=float)
    if offsets.ndim != 2 or offsets.shape[1] != 2:
        return
    record["ops"].append(
        {
            "kind": "scatter",
            "x": offsets[:, 0],
            "y": offsets[:, 1],
            "label": clean_label(kwargs.get("label", "")) or "",
        }
    )


def handle_bar(record, args, kwargs, result):
    centers = []
    heights = []
    for patch in getattr(result, "patches", []):
        centers.append(patch.get_x() + patch.get_width() / 2.0)
        heights.append(patch.get_height())
    record["ops"].append(
        {
            "kind": "bar",
            "x": array_copy(centers),
            "y": array_copy(heights),
            "label": clean_label(kwargs.get("label", "")) or "",
        }
    )


def handle_barh(record, args, kwargs, result):
    positions = []
    widths = []
    for patch in getattr(result, "patches", []):
        positions.append(patch.get_y() + patch.get_height() / 2.0)
        widths.append(patch.get_width())
    record["ops"].append(
        {
            "kind": "barh",
            "x": array_copy(widths),
            "y": array_copy(positions),
            "label": clean_label(kwargs.get("label", "")) or "",
        }
    )


def handle_boxplot(record, args, kwargs, result):
    boxes = []
    medians = result.get("medians", [])
    box_patches = result.get("boxes", [])
    caps = result.get("caps", [])
    fliers = result.get("fliers", [])
    for index, med_line in enumerate(medians):
        med_x = float(np.mean(med_line.get_xdata()))
        med_y = float(np.mean(med_line.get_ydata()))
        q1 = q3 = None
        if index < len(box_patches):
            ys = box_patches[index].get_path().vertices[:, 1]
            q1, q3 = float(ys.min()), float(ys.max())
        lo = hi = None
        if 2 * index + 1 < len(caps):
            lo = float(np.min(caps[2 * index].get_ydata()))
            hi = float(np.max(caps[2 * index + 1].get_ydata()))
        outlier_x = outlier_y = None
        if index < len(fliers):
            outlier_x = array_copy(fliers[index].get_xdata())
            outlier_y = array_copy(fliers[index].get_ydata())
        boxes.append(
            {
                "x": med_x,
                "median": med_y,
                "q1": q1,
                "q3": q3,
                "lo": lo,
                "hi": hi,
                "flier_x": outlier_x,
                "flier_y": outlier_y,
            }
        )
    labels = kwargs.get("labels")
    label = labels[0] if isinstance(labels, (list, tuple)) and labels else kwargs.get("label", "")
    record["ops"].append({"kind": "box", "boxes": boxes, "label": clean_label(label) or ""})


def handle_fill_between(record, args, kwargs, result):
    if len(args) >= 3:
        x, y1, y2 = args[0], args[1], args[2]
    else:
        return
    record["ops"].append(
        {
            "kind": "fill",
            "x": array_copy(x),
            "y1": array_copy(y1),
            "y2": array_copy(y2),
            "label": clean_label(kwargs.get("label", "")) or "",
        }
    )


def handle_vlines(record, args, kwargs, result):
    if len(args) >= 3:
        x, ymin, ymax = args
    else:
        return
    record["ops"].append(
        {
            "kind": "vseg",
            "x": array_copy(x),
            "y1": array_copy(ymin),
            "y2": array_copy(ymax),
            "label": clean_label(kwargs.get("label", "")) or "",
        }
    )


def handle_hlines(record, args, kwargs, result):
    if len(args) >= 3:
        y, xmin, xmax = args
    else:
        return
    record["ops"].append(
        {
            "kind": "hseg",
            "y": array_copy(y),
            "x1": array_copy(xmin),
            "x2": array_copy(xmax),
            "label": clean_label(kwargs.get("label", "")) or "",
        }
    )


def handle_axvline(record, args, kwargs, result):
    record["ops"].append(
        {
            "kind": "vref",
            "x": float(first_arg(args, kwargs, "x")),
            "label": clean_label(kwargs.get("label", "")) or "",
        }
    )


def handle_axhline(record, args, kwargs, result):
    record["ops"].append(
        {
            "kind": "href",
            "y": float(first_arg(args, kwargs, "y")),
            "label": clean_label(kwargs.get("label", "")) or "",
        }
    )


def handle_pcolormesh(record, args, kwargs, result):
    array = np.asarray(result.get_array())
    coordinates = result.get_coordinates()
    if coordinates.ndim == 3:
        x_edges = np.asarray(coordinates[0, :, 0], dtype=float)
        y_edges = np.asarray(coordinates[:, 0, 1], dtype=float)
        rows, cols = coordinates.shape[0] - 1, coordinates.shape[1] - 1
        if array.ndim == 1 and array.size == rows * cols:
            array = array.reshape(rows, cols)
    else:
        return
    if array.ndim != 2:
        return
    record["ops"].append(
        {
            "kind": "heat",
            "x": (x_edges[:-1] + x_edges[1:]) / 2.0,
            "y": (y_edges[:-1] + y_edges[1:]) / 2.0,
            "z": array.astype(float),
        }
    )


def handle_imshow(record, args, kwargs, result):
    array = np.asarray(result.get_array())
    if array.ndim != 2:
        return
    x0, x1, y0, y1 = result.get_extent()
    rows, cols = array.shape
    if result.origin == "upper":
        array = array[::-1]
        y0, y1 = y1, y0
    record["ops"].append(
        {
            "kind": "heat",
            "x": np.linspace(x0, x1, cols),
            "y": np.linspace(y0, y1, rows),
            "z": array.astype(float),
        }
    )


def install_wrappers():
    """安装全部绘图方法捕获器。"""
    for name, handler in (
        ("plot", handle_plot),
        ("scatter", handle_scatter),
        ("bar", handle_bar),
        ("barh", handle_barh),
        ("boxplot", handle_boxplot),
        ("fill_between", handle_fill_between),
        ("vlines", handle_vlines),
        ("hlines", handle_hlines),
        ("axvline", handle_axvline),
        ("axhline", handle_axhline),
        ("pcolormesh", handle_pcolormesh),
        ("imshow", handle_imshow),
    ):
        wrap_axes_method(name, handler)


def colorbar_axes() -> set:
    """收集颜色条坐标区，导出时跳过。"""
    skip = set()
    for record in RECORDS:
        try:
            for artist in list(record["axis"].images) + list(record["axis"].collections):
                bar = getattr(artist, "colorbar", None)
                if bar is not None and getattr(bar, "ax", None) is not None:
                    skip.add(id(bar.ax))
        except Exception:
            pass
    return skip


def op_rows(panel, op, series_hint, axis):
    """把一个绘图操作展开为主表行（子图,系列,x,y）。"""
    rows = []

    def add(series, x, y):
        xs = array_copy(x)
        ys = array_copy(y)
        if xs is None or ys is None:
            return
        count = min(len(xs), len(ys))
        for i in range(count):
            rows.append((panel, series, fmt(xs[i]), fmt(ys[i])))

    kind = op["kind"]
    if kind in ("line", "scatter"):
        add(op["label"] or series_hint, op["x"], op["y"])
    elif kind in ("bar", "barh"):
        add(op["label"] or series_hint, op["x"], op["y"])
    elif kind == "fill":
        add((op["label"] or series_hint) + " 下界", op["x"], op["y1"])
        add((op["label"] or series_hint) + " 上界", op["x"], op["y2"])
    elif kind == "vseg":
        add(op["label"] or series_hint, op["x"], op["y1"])
        add(op["label"] or series_hint, op["x"], op["y2"])
    elif kind == "hseg":
        add(op["label"] or series_hint, op["x1"], op["y"])
        add(op["label"] or series_hint, op["x2"], op["y"])
    elif kind == "vref":
        ylim = axis.get_ylim()
        add((op["label"] or series_hint) + "（竖直参考线）", [op["x"], op["x"]], [ylim[0], ylim[1]])
    elif kind == "href":
        xlim = axis.get_xlim()
        add((op["label"] or series_hint) + "（水平参考线）", [xlim[0], xlim[1]], [op["y"], op["y"]])
    elif kind == "box":
        for index, box in enumerate(op["boxes"]):
            name = (op["label"] or series_hint) + " 箱线%d" % (index + 1)
            add(name + " 中位数", [box["x"]], [box["median"]])
            if box["q1"] is not None:
                add(name + " Q1", [box["x"]], [box["q1"]])
                add(name + " Q3", [box["x"]], [box["q3"]])
            if box["lo"] is not None:
                add(name + " 下须", [box["x"]], [box["lo"]])
                add(name + " 上须", [box["x"]], [box["hi"]])
            if box["flier_x"] is not None and box["flier_y"] is not None:
                add(name + " 离群点", box["flier_x"], box["flier_y"])
    return rows


def dump_csvs(out_dir: Path, stem: str):
    """把捕获结果写入主CSV与热图CSV。"""
    skip = colorbar_axes()
    rows = []
    heat_blocks = []
    panel_no = 0
    series_no = 0
    for record in RECORDS:
        if id(record["axis"]) in skip:
            continue
        if not record["ops"]:
            continue
        panel_no += 1
        title = clean_label(record["axis"].get_title())
        panel = "子图%d" % panel_no + (("：" + title) if title else "")
        for op in record["ops"]:
            if op["kind"] == "heat":
                heat_blocks.append((panel, op))
                continue
            series_no += 1
            rows.extend(op_rows(panel, op, "系列%d" % series_no, record["axis"]))

    main_path = out_dir / (stem + ".csv")
    with main_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["子图", "系列", "x", "y"])
        writer.writerows(rows)

    for index, (panel, op) in enumerate(heat_blocks, start=1):
        heat_path = out_dir / ("%s_热图%d.csv" % (stem, index))
        x, y, z = op["x"], op["y"], op["z"]
        with heat_path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["%s 热图矩阵（行=y，列=x）" % panel] + [fmt(v) for v in x])
            for row_index in range(z.shape[0]):
                writer.writerow([fmt(y[row_index])] + [fmt(v) for v in z[row_index]])
    return len(rows), len(heat_blocks)


def run_single(script_path: Path):
    """运行单个图脚本并导出其CSV。"""
    RECORDS.clear()
    install_wrappers()
    sys.argv = [str(script_path)]
    os.environ.setdefault("MPLBACKEND", "Agg")
    runpy.run_path(str(script_path), run_name="__main__")
    rows, heats = dump_csvs(script_path.parent, script_path.stem)
    print("已导出：%d行曲线数据，%d个热图表" % (rows, heats))


def main():
    if len(sys.argv) > 1:
        run_single(Path(sys.argv[1]).resolve())
        return
    failures = []
    count = 0
    for directory in sorted(FIGURES_ROOT.iterdir()):
        if not directory.is_dir():
            continue
        scripts = sorted(directory.glob("fig*.py"))
        if not scripts:
            continue
        count += 1
        env = dict(os.environ, MPLBACKEND="Agg", PYTHONIOENCODING="utf-8")
        result = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), str(scripts[0])],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
        )
        csv_files = sorted(p.name for p in directory.glob("*.csv"))
        if result.returncode == 0:
            print("[成功] %s -> %s" % (directory.name, ", ".join(csv_files)))
        else:
            failures.append((directory.name, (result.stderr or "").strip()[-400:]))
            print("[失败] %s" % directory.name)
    print("处理图脚本数：%d，失败：%d" % (count, len(failures)))
    for name, message in failures:
        print("---- %s ----\n%s" % (name, message))


if __name__ == "__main__":
    main()
