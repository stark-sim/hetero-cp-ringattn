from copy import deepcopy
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt


BASE = Path('/Users/stark_sim/Desktop/硕士课题/开题报告')
SRC = BASE / '开题报告_新版12_沈达_面向大模型推理与强化学习后训练的异构算力负载承接研究.docx'
OUT = BASE / '开题报告_新版13_沈达_面向大模型推理与强化学习后训练的异构算力负载承接研究.docx'
IMG_DIR = Path('/Users/stark_sim/VSCodeProjects/hetero-cp-ringattn/output/imagegen')


def clone_run_format(para):
    if para.runs and para.runs[0]._r.rPr is not None:
        return deepcopy(para.runs[0]._r.rPr)
    return None


def set_text(para, text):
    ppr = deepcopy(para._p.pPr) if para._p.pPr is not None else None
    rpr = clone_run_format(para)
    para.clear()
    run = para.add_run(text)
    if ppr is not None:
        para._p.insert(0, ppr)
    if rpr is not None:
        run._r.insert(0, rpr)


def find_para(doc, marker):
    for para in doc.paragraphs:
        if marker in para.text:
            return para
    raise ValueError(f'paragraph not found: {marker}')


def make_para(doc, text, style=None, bold=False, align=None, size=None):
    para = doc.add_paragraph(style=style)
    run = para.add_run(text)
    run.bold = bold
    if size:
        run.font.size = Pt(size)
    if align is not None:
        para.alignment = align
    return para


def insert_after(anchor, element):
    anchor_node = anchor._tbl if hasattr(anchor, '_tbl') else anchor._p
    element_node = element._tbl if hasattr(element, '_tbl') else element._p
    anchor_node.addnext(element_node)
    return element


def add_caption(doc, text):
    para = make_para(doc, text, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, size=10)
    para.paragraph_format.space_before = Pt(3)
    para.paragraph_format.space_after = Pt(6)
    return para


def add_figure(doc, anchor, image_path, caption, width=6.1):
    image_para = doc.add_paragraph()
    image_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    image_para.paragraph_format.space_before = Pt(3)
    image_para.paragraph_format.space_after = Pt(0)
    image_para.add_run().add_picture(str(image_path), width=Inches(width))
    insert_after(anchor, image_para)
    cap = add_caption(doc, caption)
    insert_after(image_para, cap)
    return cap


def set_cell(cell, text, bold=False, size=9.5):
    cell.text = ''
    para = cell.paragraphs[0]
    para.alignment = WD_ALIGN_PARAGRAPH.LEFT
    para.paragraph_format.space_after = Pt(0)
    run = para.add_run(text)
    run.bold = bold
    run.font.size = Pt(size)
    for margin in ('top', 'start', 'bottom', 'end'):
        pass


def add_table_after(doc, anchor, title, headers, rows, widths):
    cap = add_caption(doc, title)
    insert_after(anchor, cap)
    table = doc.add_table(rows=1, cols=len(headers))
    table.style = 'Normal Table'
    table.autofit = True
    for i, header in enumerate(headers):
        set_cell(table.rows[0].cells[i], header, bold=True, size=9)
    for row in rows:
        cells = table.add_row().cells
        for i, value in enumerate(row):
            set_cell(cells[i], value, size=8.7)
    for row in table.rows:
        for i, width in enumerate(widths):
            row.cells[i].width = Inches(width)
    insert_after(cap, table)
    note = make_para(doc, '注：表中“新增机制”仅指本课题拟实现或验证的增量，不把参考框架已有能力写作创新。', size=9)
    note.paragraph_format.space_before = Pt(3)
    note.paragraph_format.space_after = Pt(6)
    insert_after(table, note)
    return note


def main():
    if not SRC.exists():
        raise FileNotFoundError(SRC)
    doc = Document(SRC)

    # Add macro-to-specific transitions without changing the established scope.
    p = find_para(doc, '本课题即源于对上述问题的思考与前期实践。')
    new = make_para(doc, '从研究对象的可拆分性出发，本课题将“异构承接”理解为一组带条件的系统选择：先识别负载阶段和状态依赖，再估计通信、显存、排队与恢复成本，最后决定采用请求级、阶段级或任务内协同。这样的定义既能容纳真实环境中的设备差异，也能避免将所有连接在同一网络中的加速卡都视为可以共同执行任意任务。')
    insert_after(p, new)
    p = find_para(doc, '这些工作的实际意义在于让异构资源以可预测的方式承接真实工作')
    new = make_para(doc, '因此，课题的结论将同时包含正向结果和负向边界：正向结果说明某类设备组合、上下文规模或阶段比例下存在可复现的净收益；负向边界说明通信、拖尾、版本滞后或恢复开销超过收益时应当拒绝协同。将两类结果同时写入研究结论，能够使后续系统设计具有明确的使用条件。')
    insert_after(p, new)

    p = find_para(doc, '本章按“服务组织—任务内协同—后训练工作流—弱连接协作”的顺序梳理国外研究')
    new = make_para(doc, '阅读本章时需要区分三种证据。第一种是已经在工业或开源系统中稳定存在的工程能力，例如请求路由、阶段分离、KV管理和异步流水线；第二种是论文中在受控环境下验证的并行机制，例如非均匀序列切分和低频参数同步；第三种是本课题尚需通过原型和实验确认的组合能力，即在明确通信和恢复约束下，让异构设备承接一部分真实工作。后文将按这一层次标记研究空缺，避免把“可借鉴”误写成“已解决”。')
    insert_after(p, new)
    p = find_para(doc, '由此，本课题的两条路线可共享的分析问题可具体表述为')
    new = make_para(doc, '这一综述还给出一个重要的节奏判断：研究不应从“如何把更多设备加入任务”开始，而应从“不加入会损失什么、加入后新增什么代价”开始。推理线先验证长上下文是否存在显存墙，再决定是否打开HCP；RL线先确认阶段边界和版本约束，再决定是否把弹性设备纳入工作流。两条路线均以能够被实验拒绝的准入条件作为起点。')
    insert_after(p, new)

    # Add overview figure and a compact research-plan matrix at the beginning of Chapter 3.
    p = find_para(doc, '两条路线均从相同的分析维度出发')
    overview = IMG_DIR / 'heterogeneous-two-research-directions-overview-v4-labeled.png'
    if overview.exists():
        cap = add_figure(doc, p, overview, '图1  两条独立研究路线及其异构负载承接边界', width=6.0)
        p = cap
    rows = [
        ('长上下文Prefill', '输入长度、KV规模、显存、P2P带宽/时延、TTFT约束', 'HCP准入；capacity-aware非对称seq_chunk_len/block_size；K/V P2P ring；超时和容量变化回退', '单设备、同构CP、请求/PD路由、固定均分HCP；TTFT、尾延迟、显存峰值、网络字节数、回退率'),
        ('LLM-RL后训练阶段', '阶段输入输出、后端兼容性、版本、队列、恢复窗口', '阶段能力合同；rollout/奖励评估/数据处理异构放置；版本与样本完整性检查；节点退出回退', '固定同位、静态阶段放置、异步流水线、动态阶段承接；有效样本吞吐、样本有效率、版本滞后、恢复时间'),
    ]
    note = add_table_after(doc, p, '表1  两条研究路线的方案闭环', ['研究路线', '输入与约束', '本课题新增机制', '基线与主要评价'], rows, [1.0, 1.7, 2.1, 1.6])

    # Route-specific figures stay inside their own sections.
    p = find_para(doc, '本研究只把长上下文Prefill作为任务内异构协同的验证对象')
    hcp = IMG_DIR / 'heterogeneous-hcp-prefill-detail-v3.png'
    if hcp.exists():
        add_figure(doc, p, hcp, '图2  长上下文Prefill中HCP的非对称切分、K/V环传递与准入回退', width=5.9)
    p = find_para(doc, '本研究以LLM-RL后训练工作流为对象')
    rl = IMG_DIR / 'heterogeneous-llm-rl-stage-admission-v3.png'
    if rl.exists():
        add_figure(doc, p, rl, '图3  LLM-RL后训练的阶段级异构承接与弹性回退', width=5.9)

    # Add a capability boundary table in the completed-work chapter.
    p = find_para(doc, '（1）形成 hetero-cp-ringattn 开源代码库和异构通信原型')
    rows = [
        ('vLLM / Dynamo类框架', 'API、请求路由、Prefill/Decode分离、KV生命周期、常规worker执行', '不重写服务框架；增加HCP候选计划的调用接口与回退状态'),
        ('HCP原型', '非均匀seq_chunk_len、block_size、K/V P2P ring、online softmax协议基础', '补充设备/链路画像、准入代价估计、容量感知切分和服务适配验证'),
        ('HetRL / Prime RL', '阶段放置、异步rollout、训练/评估工作流组织', '组合阶段能力合同、版本/队列约束、样本完整性与动态回退'),
        ('Prime DiLoCo', '弹性设备网格、低频同步、异步检查点和恢复参考', '在LLM-RL阶段工作流中验证节点加入/退出、链路降级和恢复窗口'),
    ]
    add_table_after(doc, p, '表2  参考框架已有能力与本课题新增机制', ['参考对象', '已有能力（作为基线或接口）', '本课题新增或待验证内容'], rows, [1.3, 3.0, 2.1])

    # Make key section headings visually distinct while preserving document fonts.
    for marker in ['1.  课题来源及研究的背景和意义', '2. 国内外在该方向的研究现状及分析', '3. 主要研究内容及实施方案', '4. 预 期 达 到 的 目 标', '5. 已 完 成 的 研 究 工 作 与 进 度 安 排', '6.  为完成课题已具备和所需的条件和经费', '7.  预计研究过程中可能遇到的困难和问题，以及解决的措施', '8.  主要参考文献']:
        try:
            para = find_para(doc, marker)
            if para.runs:
                para.runs[0].bold = True
        except ValueError:
            pass

    doc.save(OUT)
    print(f'wrote {OUT}')


if __name__ == '__main__':
    main()
