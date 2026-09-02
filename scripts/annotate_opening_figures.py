#!/usr/bin/env python3
"""Add deterministic, compact Chinese labels to the GPT Image 2 figure references."""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "output" / "imagegen"
FONT = "/System/Library/Fonts/STHeiti Medium.ttc"
INK = (38, 46, 54, 255)
ACCENT = (71, 106, 149, 255)
WHITE = (255, 255, 255, 232)


def font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONT, size=size, index=0)


def label(layer: Image.Image, xy: tuple[int, int], text: str, size: int = 30,
          color=INK, anchor: str = "mm", pad: int = 8, border=None) -> None:
    draw = ImageDraw.Draw(layer)
    fnt = font(size)
    box = draw.textbbox((0, 0), text, font=fnt, anchor=anchor)
    if anchor == "mm":
        width = box[2] - box[0] + pad * 2
        height = box[3] - box[1] + pad * 2
        x, y = xy
        rect = (x - width // 2, y - height // 2, x + width // 2, y + height // 2)
    else:
        x, y = xy
        rect = (x - pad, y - pad, x + box[2] - box[0] + pad, y + box[3] - box[1] + pad)
    draw.rounded_rectangle(rect, radius=8, fill=WHITE, outline=border, width=2 if border else 1)
    draw.text(xy, text, font=fnt, fill=color, anchor=anchor)


def save_with_labels(source: str, target: str, annotate) -> None:
    image = Image.open(FIG_DIR / source).convert("RGBA")
    layer = Image.new("RGBA", image.size, (255, 255, 255, 0))
    annotate(layer)
    image = Image.alpha_composite(image, layer).convert("RGB")
    image.save(FIG_DIR / target, format="PNG", optimize=True)


def overview(layer: Image.Image) -> None:
    label(layer, (255, 205), "异构资源池", 34, border=INK)
    label(layer, (682, 427), "能力感知准入", 31, border=ACCENT)
    label(layer, (1148, 225), "长上下文 Prefill", 30, border=ACCENT)
    label(layer, (1565, 225), "异构协同执行", 28, border=INK)
    label(layer, (1138, 625), "LLM-RL 后训练阶段", 30, border=ACCENT)
    label(layer, (1560, 650), "Rollout / 奖励 / 评估", 26, border=INK)
    label(layer, (1862, 448), "服务输出", 29, border=INK)
    label(layer, (1160, 470), "不均分 CP + K/V 交换", 23, color=ACCENT, border=ACCENT)
    label(layer, (1160, 880), "阶段级异构承接", 23, color=ACCENT, border=ACCENT)


def hcp(layer: Image.Image) -> None:
    label(layer, (1024, 84), "长上下文序列：不均匀切分", 33, border=INK)
    label(layer, (1060, 195), "大分片", 27, color=ACCENT, border=ACCENT)
    label(layer, (630, 790), "小分片", 27, color=ACCENT, border=ACCENT)
    label(layer, (1460, 790), "中分片", 27, color=ACCENT, border=ACCENT)
    label(layer, (1030, 555), "K/V P2P Ring", 30, color=ACCENT, border=ACCENT)
    label(layer, (1320, 1050), "准入失败 / 回退", 26, border=INK)


def rl(layer: Image.Image) -> None:
    label(layer, (265, 392), "Rollout", 31, border=INK)
    label(layer, (760, 392), "奖励 / 评分", 30, border=INK)
    label(layer, (1260, 392), "评估与数据处理", 28, border=INK)
    label(layer, (1760, 392), "策略更新", 31, border=INK)
    label(layer, (1760, 212), "局部高带宽设备组", 29, color=ACCENT, border=ACCENT)
    label(layer, (760, 690), "前三阶段：可异构承接", 28, color=ACCENT, border=ACCENT)
    label(layer, (760, 1030), "异构设备池", 27, border=INK)


def separate_overview(layer: Image.Image) -> None:
    label(layer, (510, 125), "方向一：长上下文推理", 38, color=ACCENT, border=ACCENT)
    label(layer, (510, 172), "任务内异构协同", 25, border=INK)
    label(layer, (1505, 125), "方向二：LLM-RL 后训练", 38, color=ACCENT, border=ACCENT)
    label(layer, (1505, 172), "阶段级异构承接", 25, border=INK)
    label(layer, (132, 255), "异构设备池", 26, border=INK)
    label(layer, (355, 255), "HCP 准入判断", 25, border=ACCENT)
    label(layer, (610, 255), "长上下文 Prefill", 25, border=INK)
    label(layer, (890, 255), "非对称 HCP", 26, color=ACCENT, border=ACCENT)
    label(layer, (890, 820), "不均匀切分 + K/V Ring", 22, color=ACCENT, border=ACCENT)
    label(layer, (1190, 255), "异构设备池", 26, border=INK)
    label(layer, (1435, 255), "阶段能力合同", 25, border=ACCENT)
    label(layer, (1665, 255), "Rollout / 奖励 / 评估", 23, border=INK)
    label(layer, (1900, 255), "局部高带宽\n策略更新", 24, color=ACCENT, border=ACCENT)
    label(layer, (1540, 820), "版本、队列与弹性回退", 22, color=ACCENT, border=ACCENT)


def main() -> None:
    save_with_labels("heterogeneous-load-admission-overview-v2.png", "heterogeneous-load-admission-overview-v3.png", overview)
    save_with_labels("heterogeneous-hcp-prefill-detail-v2.png", "heterogeneous-hcp-prefill-detail-v3.png", hcp)
    save_with_labels("heterogeneous-llm-rl-stage-admission-v2.png", "heterogeneous-llm-rl-stage-admission-v3.png", rl)
    save_with_labels(
        "heterogeneous-two-research-directions-overview-v4.png",
        "heterogeneous-two-research-directions-overview-v4-labeled.png",
        separate_overview,
    )


if __name__ == "__main__":
    main()
