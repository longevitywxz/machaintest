from __future__ import annotations

from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "report"
OUTPUT_DIR = ROOT / "outputs"
FONT_PATH = Path(r"C:\Windows\Fonts\simhei.ttf")


def register_fonts() -> str:
    if FONT_PATH.exists():
        pdfmetrics.registerFont(TTFont("SimHei", str(FONT_PATH)))
        return "SimHei"
    return "Helvetica"


def add_paragraph(story: list, text: str, style: ParagraphStyle, space: float = 0.25) -> None:
    story.append(Paragraph(text, style))
    story.append(Spacer(1, space * cm))


def build_metric_table(font_name: str) -> Table:
    summary = pd.read_csv(OUTPUT_DIR / "metrics_summary.csv")
    data = [["Model", "Horizon", "MSE mean", "MSE std", "MAE mean", "MAE std"]]
    for row in summary.itertuples(index=False):
        data.append(
            [
                row.model,
                str(row.horizon),
                f"{row.mse_mean:.2f}",
                f"{row.mse_std:.2f}",
                f"{row.mae_mean:.2f}",
                f"{row.mae_std:.2f}",
            ]
        )
    table = Table(data, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, -1), font_name),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#d9eaf7")),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    return table


def main() -> None:
    REPORT_DIR.mkdir(exist_ok=True)
    font_name = register_fonts()
    styles = getSampleStyleSheet()
    title = ParagraphStyle("TitleCN", parent=styles["Title"], fontName=font_name, fontSize=18, leading=24)
    h1 = ParagraphStyle("H1CN", parent=styles["Heading1"], fontName=font_name, fontSize=13, leading=18, spaceBefore=8)
    h2 = ParagraphStyle("H2CN", parent=styles["Heading2"], fontName=font_name, fontSize=11, leading=16, spaceBefore=6)
    body = ParagraphStyle("BodyCN", parent=styles["BodyText"], fontName=font_name, fontSize=9.5, leading=15)

    story: list = []
    add_paragraph(story, "家庭电力消耗多变量时间序列预测实验报告", title)
    add_paragraph(story, "作者信息", h1)
    add_paragraph(story, "姓名：魏宪正；学号：20255227057；团队人数：1 人。", body)
    add_paragraph(story, "所属研究领域：大模型、RAG、Agent、多模态。", body)
    add_paragraph(story, "贡献分工：本人独立完成数据处理、模型实现、实验训练、结果分析、图表绘制和报告撰写，贡献比例 100%。", body)
    add_paragraph(story, "GitHub 代码链接：https://github.com/longevitywxz/machaintest", body)
    add_paragraph(story, "1. 问题介绍", h1)
    add_paragraph(
        story,
        "本项目研究家庭电力消耗预测问题。数据来自 UCI Individual Household Electric Power Consumption，"
        "原始记录以分钟为粒度。实验将分钟数据汇总为日级数据：功率与子表能耗按天求和，"
        "电压和电流按天求平均，并补充 sub_metering_remainder、日期周期特征和 data.gouv / Meteo-France 92 省月度天气变量。",
        body,
    )
    add_paragraph(
        story,
        "预测任务为使用过去 90 天的多变量序列预测未来总有功功率曲线，分别设置 90 天短期预测和 "
        "365 天长期预测。两种预测长度分别训练模型，评价指标为 MSE 和 MAE。每个模型重复 5 个随机种子实验。",
        body,
    )

    add_paragraph(story, "2. 模型", h1)
    add_paragraph(story, "2.1 LSTM", h2)
    add_paragraph(story, "LSTM 使用最后一层隐藏状态表示历史用电模式，再通过全连接层直接输出未来 horizon 天预测序列。", body)
    add_paragraph(story, "2.2 Transformer", h2)
    add_paragraph(story, "Transformer 使用多头自注意力编码 90 天历史序列，并以最后时间步表示输出未来曲线。", body)
    add_paragraph(story, "2.3 改进模型：CNN-Transformer", h2)
    add_paragraph(
        story,
        "CNN-Transformer 先用一维卷积提取局部用电波动，再输入 Transformer 建模长期依赖，最后使用均值池化与门控融合输出预测。",
        body,
    )
    add_paragraph(story, "结构伪代码", h2)
    pseudocode = (
        "Input X in R^(90 x d); "
        "Z = GELU(Conv1D_k5(X)); "
        "Z = GELU(Conv1D_k3(Z)); "
        "E = TransformerEncoder(PositionalEncoding(Z)); "
        "c = MeanPool(E) * Sigmoid(W_g E_last + b_g); "
        "y_hat = Linear(LayerNorm(c))."
    )
    add_paragraph(story, pseudocode, body)
    add_paragraph(story, "结构流程：90 天多变量输入 -> 1D CNN 局部滤波 -> 位置编码 -> Transformer 编码 -> 均值池化与末状态门控 -> 90/365 天预测。", body)

    add_paragraph(story, "3. 结果与分析", h1)
    story.append(build_metric_table(font_name))
    story.append(Spacer(1, 0.4 * cm))
    add_paragraph(
        story,
        "从结果看，90 天预测比 365 天预测更容易保持趋势和幅值稳定；长期预测需要跨季节建模，误差更大。"
        "CNN-Transformer 在 365 天长期预测上优于 LSTM，并接近 Transformer；在 90 天预测中 MSE 接近 Transformer，但 MAE 略高。"
        "这说明卷积局部特征有助于长期趋势建模，但也可能削弱短期尖峰拟合。",
        body,
    )
    add_paragraph(
        story,
        "CNN-Transformer 的优势来自卷积预先整合邻近日子的局部模式，以及均值池化与最后状态门控对整体趋势和最近状态的融合。"
        "不足是月度天气变量粒度较粗、样本数量有限，模型在短期任务上容易过平滑。后续可加入逐日气象、节假日和异常用电标记。",
        body,
    )

    figures = [
        ("LSTM 90-day", "lstm_90.png"),
        ("Transformer 90-day", "transformer_90.png"),
        ("CNN-Transformer 90-day", "cnn_transformer_90.png"),
        ("LSTM 365-day", "lstm_365.png"),
        ("Transformer 365-day", "transformer_365.png"),
        ("CNN-Transformer 365-day", "cnn_transformer_365.png"),
    ]
    for idx, (caption, filename) in enumerate(figures):
        if idx % 2 == 0:
            story.append(PageBreak())
        add_paragraph(story, caption, h2, space=0.1)
        story.append(Image(str(OUTPUT_DIR / "figures" / filename), width=17 * cm, height=6.2 * cm))
        story.append(Spacer(1, 0.35 * cm))

    story.append(PageBreak())
    add_paragraph(story, "4. 讨论", h1)
    add_paragraph(
        story,
        "本实验采用直接多步预测，避免递归误差累积，但要求模型一次性学习完整未来曲线。已加入月度天气变量，"
        "后续可加入更细粒度的逐日气象、节假日特征，或先分解趋势与季节项后预测残差。报告撰写使用 ChatGPT/Codex 辅助整理文字和代码结构。",
        body,
    )
    add_paragraph(story, "参考文献", h1)
    add_paragraph(story, "[1] UCI Machine Learning Repository. Individual household electric power consumption.", body)
    add_paragraph(story, "[2] data.gouv / Meteo-France. Donnees climatologiques de base mensuelles.", body)
    add_paragraph(story, "[3] Vaswani, A. et al. Attention Is All You Need. NeurIPS, 2017.", body)
    add_paragraph(story, "[4] Hochreiter, S., Schmidhuber, J. Long Short-Term Memory. Neural Computation, 1997.", body)
    add_paragraph(story, "[5] 课程作业说明《2026年专硕机器学习课程项目》。", body)

    doc = SimpleDocTemplate(
        str(REPORT_DIR / "report.pdf"),
        pagesize=A4,
        rightMargin=1.6 * cm,
        leftMargin=1.6 * cm,
        topMargin=1.6 * cm,
        bottomMargin=1.6 * cm,
    )
    doc.build(story)
    print(REPORT_DIR / "report.pdf")


if __name__ == "__main__":
    main()
