from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "report"
OUTPUT_DIR = ROOT / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"
PDF_PATH = REPORT_DIR / "report.pdf"
FINAL_PDF_PATH = REPORT_DIR / "20255227057-魏宪正.pdf"
FONT_PATH = Path(r"C:\Windows\Fonts\simhei.ttf")
FONT_NAME = "SimHei"

TITLE = "家庭电力消耗多变量时间序列预测实验报告"
GITHUB_URL = "https://github.com/longevitywxz/machaintest"

ACCENT = colors.HexColor("#24577A")
ACCENT_LIGHT = colors.HexColor("#EAF3F8")
TEXT = colors.HexColor("#1F2933")
MUTED = colors.HexColor("#52606D")
GRID = colors.HexColor("#CBD5E1")
CODE_BG = colors.HexColor("#F6F8FA")


@dataclass(frozen=True)
class ReportStyles:
    title: ParagraphStyle
    subtitle: ParagraphStyle
    h1: ParagraphStyle
    h2: ParagraphStyle
    body: ParagraphStyle
    small: ParagraphStyle
    code: ParagraphStyle
    caption: ParagraphStyle


def register_fonts() -> str:
    if FONT_PATH.exists():
        pdfmetrics.registerFont(TTFont(FONT_NAME, str(FONT_PATH)))
        return FONT_NAME
    return "Helvetica"


def build_styles(font_name: str) -> ReportStyles:
    base = getSampleStyleSheet()
    return ReportStyles(
        title=ParagraphStyle(
            "TitleCN",
            parent=base["Title"],
            fontName=font_name,
            fontSize=20,
            leading=28,
            textColor=TEXT,
            alignment=TA_CENTER,
            spaceAfter=0.25 * cm,
        ),
        subtitle=ParagraphStyle(
            "SubtitleCN",
            parent=base["BodyText"],
            fontName=font_name,
            fontSize=9.5,
            leading=15,
            textColor=MUTED,
            alignment=TA_CENTER,
        ),
        h1=ParagraphStyle(
            "H1CN",
            parent=base["Heading1"],
            fontName=font_name,
            fontSize=13.5,
            leading=18,
            textColor=ACCENT,
            spaceBefore=0.25 * cm,
            spaceAfter=0.16 * cm,
        ),
        h2=ParagraphStyle(
            "H2CN",
            parent=base["Heading2"],
            fontName=font_name,
            fontSize=11,
            leading=15,
            textColor=TEXT,
            spaceBefore=0.18 * cm,
            spaceAfter=0.10 * cm,
        ),
        body=ParagraphStyle(
            "BodyCN",
            parent=base["BodyText"],
            fontName=font_name,
            fontSize=9.4,
            leading=15.2,
            textColor=TEXT,
            alignment=TA_LEFT,
            firstLineIndent=0.55 * cm,
        ),
        small=ParagraphStyle(
            "SmallCN",
            parent=base["BodyText"],
            fontName=font_name,
            fontSize=8.4,
            leading=12.5,
            textColor=MUTED,
        ),
        code=ParagraphStyle(
            "CodeCN",
            parent=base["Code"],
            fontName="Courier",
            fontSize=7.8,
            leading=10.5,
            textColor=colors.HexColor("#111827"),
            leftIndent=0,
            firstLineIndent=0,
        ),
        caption=ParagraphStyle(
            "CaptionCN",
            parent=base["BodyText"],
            fontName=font_name,
            fontSize=8.5,
            leading=12,
            textColor=MUTED,
            alignment=TA_CENTER,
            spaceBefore=0.08 * cm,
        ),
    )


def paragraph(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(text, style)


def spacer(height_cm: float = 0.18) -> Spacer:
    return Spacer(1, height_cm * cm)


def section(story: list, title: str, styles: ReportStyles) -> None:
    story.append(paragraph(title, styles.h1))


def subsection(story: list, title: str, styles: ReportStyles) -> None:
    story.append(paragraph(title, styles.h2))


def body(story: list, text: str, styles: ReportStyles) -> None:
    story.append(paragraph(text, styles.body))
    story.append(spacer(0.12))


def make_info_table(font_name: str) -> Table:
    rows = [
        ["姓名", "魏宪正", "学号", "20255227057"],
        ["团队人数", "1 人", "贡献比例", "100%"],
        ["研究领域", "大模型、RAG、Agent、多模态", "GitHub", GITHUB_URL],
    ]
    table = Table(rows, colWidths=[2.0 * cm, 5.3 * cm, 2.1 * cm, 7.0 * cm])
    table.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, -1), font_name),
                ("FONTSIZE", (0, 0), (-1, -1), 8.7),
                ("TEXTCOLOR", (0, 0), (-1, -1), TEXT),
                ("BACKGROUND", (0, 0), (0, -1), ACCENT_LIGHT),
                ("BACKGROUND", (2, 0), (2, -1), ACCENT_LIGHT),
                ("TEXTCOLOR", (0, 0), (0, -1), ACCENT),
                ("TEXTCOLOR", (2, 0), (2, -1), ACCENT),
                ("GRID", (0, 0), (-1, -1), 0.35, GRID),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    return table


def make_metric_table(font_name: str) -> Table:
    summary = pd.read_csv(OUTPUT_DIR / "metrics_summary.csv")
    header = ["模型", "预测长度", "MSE 均值", "MSE 标准差", "MAE 均值", "MAE 标准差"]
    rows = [header]
    for row in summary.itertuples(index=False):
        rows.append(
            [
                row.model,
                f"{row.horizon}",
                f"{row.mse_mean:.2f}",
                f"{row.mse_std:.2f}",
                f"{row.mae_mean:.2f}",
                f"{row.mae_std:.2f}",
            ]
        )

    table = Table(rows, colWidths=[4.0 * cm, 2.2 * cm, 2.55 * cm, 2.55 * cm, 2.55 * cm, 2.55 * cm])
    table.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, -1), font_name),
                ("BACKGROUND", (0, 0), (-1, 0), ACCENT),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("BACKGROUND", (0, 1), (-1, -1), colors.white),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
                ("GRID", (0, 0), (-1, -1), 0.35, GRID),
                ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
                ("ALIGN", (0, 0), (-1, 0), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("FONTSIZE", (0, 0), (-1, -1), 8.2),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    return table


def make_code_block(lines: list[str], styles: ReportStyles) -> Table:
    escaped = "<br/>".join(line.replace(" ", "&nbsp;") for line in lines)
    table = Table([[paragraph(escaped, styles.code)]], colWidths=[16.6 * cm])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), CODE_BG),
                ("BOX", (0, 0), (-1, -1), 0.45, GRID),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ]
        )
    )
    return table


def add_figure(story: list, title: str, filename: str, styles: ReportStyles) -> None:
    image_path = FIGURE_DIR / filename
    block = [
        paragraph(title, styles.h2),
        Image(str(image_path), width=16.6 * cm, height=6.0 * cm),
        paragraph("蓝色为真实值 Ground Truth，橙色为模型预测曲线。", styles.caption),
    ]
    story.append(KeepTogether(block))
    story.append(spacer(0.25))


def draw_header_footer(canvas, doc) -> None:
    canvas.saveState()
    width, height = A4
    canvas.setStrokeColor(GRID)
    canvas.setLineWidth(0.35)
    canvas.line(doc.leftMargin, height - 1.18 * cm, width - doc.rightMargin, height - 1.18 * cm)
    canvas.setFont(FONT_NAME if FONT_PATH.exists() else "Helvetica", 8)
    canvas.setFillColor(MUTED)
    canvas.drawString(doc.leftMargin, height - 0.88 * cm, "机器学习课程项目")
    canvas.drawRightString(width - doc.rightMargin, height - 0.88 * cm, "家庭电力消耗预测")
    canvas.line(doc.leftMargin, 1.08 * cm, width - doc.rightMargin, 1.08 * cm)
    canvas.drawCentredString(width / 2, 0.72 * cm, f"{doc.page}")
    canvas.restoreState()


def add_cover(story: list, styles: ReportStyles, font_name: str) -> None:
    story.append(spacer(0.55))
    story.append(paragraph(TITLE, styles.title))
    story.append(paragraph("基于 LSTM、Transformer 与 CNN-Transformer 的多变量时间序列预测", styles.subtitle))
    story.append(spacer(0.45))
    story.append(make_info_table(font_name))
    story.append(spacer(0.22))
    body(
        story,
        "本人独立完成数据处理、模型实现、实验训练、结果分析、图表绘制和报告撰写。完整代码、实验结果与最终 PDF 均已提交到 GitHub 仓库。",
        styles,
    )


def add_intro(story: list, styles: ReportStyles) -> None:
    section(story, "1. 问题介绍", styles)
    body(
        story,
        "本项目研究家庭电力消耗预测问题。数据来自 UCI Individual Household Electric Power Consumption，原始记录以分钟为粒度，包含全屋有功功率、无功功率、电压、电流以及三个子表能耗。",
        styles,
    )
    body(
        story,
        "实验将分钟数据汇总为日级数据：global_active_power、global_reactive_power 与三个 sub_metering 变量按天求和，voltage 和 global_intensity 按天求平均，并补充 sub_metering_remainder、日期周期特征以及 data.gouv / Meteo-France 92 省月度天气变量。",
        styles,
    )
    body(
        story,
        "预测任务为使用过去 90 天的多变量序列预测未来总有功功率曲线，分别设置 90 天短期预测和 365 天长期预测。两种预测长度分别训练模型，评价指标为 MSE 和 MAE。每个模型重复 5 个随机种子实验，并报告均值和标准差。",
        styles,
    )


def add_models(story: list, styles: ReportStyles) -> None:
    section(story, "2. 模型", styles)
    subsection(story, "2.1 LSTM", styles)
    body(story, "LSTM 将 90 天多变量序列按时间步输入循环网络，使用最后一层隐藏状态作为历史用电模式的表示，再通过全连接层直接输出未来 horizon 天预测序列。", styles)
    subsection(story, "2.2 Transformer", styles)
    body(story, "Transformer 首先将每日特征投影到隐藏维度并加入正弦位置编码，随后使用多头自注意力编码历史序列，最后取末时间步表示输出未来曲线。", styles)
    subsection(story, "2.3 改进模型：CNN-Transformer", styles)
    body(story, "CNN-Transformer 先用一维卷积提取局部用电波动，再输入 Transformer 建模长期依赖，最后使用均值池化与末状态门控融合输出预测。", styles)
    story.append(
        make_code_block(
            [
                "Input: X in R^(90 x d), horizon H in {90, 365}",
                "Z = GELU(Conv1D(X, kernel=5))",
                "Z = GELU(Conv1D(Z, kernel=3))",
                "E = TransformerEncoder(PositionalEncoding(Z))",
                "c_mean = MeanPool(E)",
                "c_last = E[-1]",
                "gate = Sigmoid(W_g c_last + b_g)",
                "y_hat = Linear(LayerNorm(c_mean * gate))",
                "Output: y_hat in R^H",
            ],
            styles,
        )
    )
    story.append(spacer(0.16))
    story.append(
        make_code_block(
            [
                "90-day multivariate input",
                "    -> 1D CNN local filters",
                "    -> positional encoding",
                "    -> Transformer encoder",
                "    -> mean pooling + last-state gate",
                "    -> 90-day / 365-day power forecast",
            ],
            styles,
        )
    )
    story.append(spacer(0.2))


def add_results(story: list, styles: ReportStyles, font_name: str) -> None:
    section(story, "3. 结果与分析", styles)
    story.append(make_metric_table(font_name))
    story.append(spacer(0.28))
    body(
        story,
        "从结果看，90 天预测比 365 天预测更容易保持趋势和幅值稳定；长期预测需要跨季节建模，误差更大。CNN-Transformer 在 365 天长期预测上优于 LSTM，并接近 Transformer；在 90 天预测中 MSE 接近 Transformer，但 MAE 略高。",
        styles,
    )
    body(
        story,
        "该现象说明卷积层提取的局部平滑特征有助于长期趋势建模，但也可能削弱短期尖峰拟合。家庭用电序列存在较强随机性，直接用日级序列预测 365 天会使模型更倾向于学习平滑趋势，而难以恢复真实曲线中的高频波动。",
        styles,
    )
    body(
        story,
        "CNN-Transformer 的优势来自卷积预先整合邻近日子的局部模式，以及均值池化与最后状态门控对整体趋势和最近状态的融合。不足是月度天气变量粒度较粗、样本数量有限，模型在短期任务上容易过平滑。",
        styles,
    )


def add_figures(story: list, styles: ReportStyles) -> None:
    figures = [
        ("图 1  LSTM 90 天预测", "lstm_90.png"),
        ("图 2  Transformer 90 天预测", "transformer_90.png"),
        ("图 3  CNN-Transformer 90 天预测", "cnn_transformer_90.png"),
        ("图 4  LSTM 365 天预测", "lstm_365.png"),
        ("图 5  Transformer 365 天预测", "transformer_365.png"),
        ("图 6  CNN-Transformer 365 天预测", "cnn_transformer_365.png"),
    ]
    for index, (title, filename) in enumerate(figures):
        if index % 2 == 0:
            story.append(PageBreak())
        add_figure(story, title, filename, styles)


def add_discussion(story: list, styles: ReportStyles) -> None:
    story.append(PageBreak())
    section(story, "4. 讨论", styles)
    body(
        story,
        "本实验采用直接多步预测，避免递归误差累积，但要求模型一次性学习完整未来曲线。日期周期特征提供星期和月份信息，天气特征提供降水和雾等外部环境信息，有助于模型拟合生活规律、季节性变化和气候影响。",
        styles,
    )
    body(
        story,
        "后续改进可以加入更细粒度的逐日气象、节假日和异常用电标记，也可以采用趋势-残差分解或多尺度卷积来改善尖峰预测。本报告撰写过程中使用了 ChatGPT/Codex 辅助整理文字和代码结构；模型设计、实验运行与结果分析以仓库中的可复现实验输出为准。",
        styles,
    )
    section(story, "参考文献", styles)
    references = [
        "[1] UCI Machine Learning Repository. Individual household electric power consumption.",
        "[2] data.gouv / Meteo-France. Donnees climatologiques de base mensuelles.",
        "[3] Vaswani, A. et al. Attention Is All You Need. NeurIPS, 2017.",
        "[4] Hochreiter, S., Schmidhuber, J. Long Short-Term Memory. Neural Computation, 1997.",
        "[5] 课程作业说明《2026年专硕机器学习课程项目》。",
    ]
    for item in references:
        story.append(paragraph(item, styles.small))
        story.append(spacer(0.06))


def build_pdf() -> None:
    REPORT_DIR.mkdir(exist_ok=True)
    font_name = register_fonts()
    styles = build_styles(font_name)
    story: list = []
    add_cover(story, styles, font_name)
    add_intro(story, styles)
    add_models(story, styles)
    add_results(story, styles, font_name)
    add_figures(story, styles)
    add_discussion(story, styles)

    doc = SimpleDocTemplate(
        str(PDF_PATH),
        pagesize=A4,
        rightMargin=1.55 * cm,
        leftMargin=1.55 * cm,
        topMargin=1.45 * cm,
        bottomMargin=1.35 * cm,
    )
    doc.build(story, onFirstPage=draw_header_footer, onLaterPages=draw_header_footer)
    FINAL_PDF_PATH.write_bytes(PDF_PATH.read_bytes())
    print(PDF_PATH)
    print(FINAL_PDF_PATH)


if __name__ == "__main__":
    build_pdf()
