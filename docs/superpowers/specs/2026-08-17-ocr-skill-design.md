# OCR Skill 设计文档：给纯文本模型装上"眼睛"

- 日期：2026-08-17
- 状态：已与用户逐节确认
- 目标运行时：DeepSeek Harness（DSH）技能系统，`~/.agents/skills/ocr`（符号链接到本仓库）

## 1. 需求分析

用轻量高准确率的 OCR 模型做一个 skill，让不支持多模态的纯文本 LLM（如 deepseek v4 flash）也能"看懂"图片。

能力拆解为三层：

| 层级 | 需求 | 说明 |
|---|---|---|
| L1 | 文字提取 | 截图/网页/聊天记录/扫描件 → 阅读顺序文本 |
| L2 | 表格结构化 | 表格 → TSV/CSV，供模型做数据分析 |
| L3 | 布局/结构理解 + 设计审查 | 多页 PDF、UI 界面布局（标题/按钮/导航层级、字号层级）、对比度/字号/对齐/配色审查 |

**核心洞察**：OCR 只负责"廉价地看到"（文字 + 坐标 + 置信度 + 像素事实），"理解"与"判断"交给纯文本 LLM——它擅长从坐标推断布局。识别（ocr.py）与审查（analyze.py）解耦，按需调用，控制 token。

### 用户确认的决策

- 场景：屏幕截图/网页/聊天记录、表格数据、多页 PDF + 布局/UI 结构理解 + 设计审查
- 表格：需要结构化输出（TSV）
- 语言：中英为主（PP-OCRv6 多语模型天然覆盖）
- 审查检查项：文字对比度、字号层级一致性、对齐/间距一致性、页面配色统计（四项全要）
- 架构：方案 A（单体 Python CLI + 多模式输出），否决常驻服务（B），VLM 层（C）仅留接口不做

## 2. 引擎选型：PP-OCRv6 via RapidOCR（rapidocr>=3.9.0）

### 2.1 模型规格（源码实测确认）

PP-OCRv6 在 rapidocr 中是多语种统一模型（`lang_type` 不影响选择），det/rec 各三个规格：

| 规格 | det | rec | 定位 |
|---|---|---|---|
| `tiny` | multi_PP-OCRv6_det_tiny | multi_PP-OCRv6_rec_tiny | 最快，~1.5M 参数级 |
| `small` | multi_PP-OCRv6_det_small | multi_PP-OCRv6_rec_small | 均衡 |
| `medium` | multi_PP-OCRv6_det_medium | multi_PP-OCRv6_rec_medium | **默认（推荐）**，~34.5M 参数级，准确率最高 |

CLI 提供 `--model-type tiny|small|medium`（默认 medium）与 `--fast` 快捷参数（等价 `--model-type small`）。

**事实基准（rapidocr 3.9.2 `config.yaml` 实测）**：
- 出厂默认即 `PP-OCRv6` + `small`——我们显式锁定 `medium` 是**精度决策**，不是修正默认 bug
- **方向分类（cls）无 v6 版本**：官方默认沿用 `PP-OCRv4` mobile cls 模型，正确做法是不改动它
- 模型下载源：ModelScope（`https://www.modelscope.cn/models/RapidAI/RapidOCR`），SHA256 校验，国内可直连

### 2.2 引擎配置

- 推理引擎：`ONNXRUNTIME` + **CPU Provider**。**禁用 CoreML**：RapidOCR 官方 M2 实测（https://raw.githubusercontent.com/RapidAI/RapidOCRDocs/main/docs/blog/posts/inference_engine/compare_coreml_cpu_provider_perf.md）显示 CoreML 对 PP-OCR 系列模型优化不足——检测慢 3.5–6.8 倍、识别慢 3.2–14 倍（精度相同）。M3 同理。
- 线程：保持默认 `-1`（由 onnxruntime 自动决定，通常为物理核心数）。
- **显式锁定写法**（3.9.x API 为 `RapidOCR(config_path=None, params: dict)`，点分键 + 枚举值，非关键字参数）：
  ```python
  from rapidocr import RapidOCR, ModelType, OCRVersion
  engine = RapidOCR(params={
      "Det.ocr_version": OCRVersion.PPOCRV6,
      "Det.model_type": ModelType.MEDIUM,
      "Rec.ocr_version": OCRVersion.PPOCRV6,
      "Rec.model_type": ModelType.MEDIUM,
      "Global.log_level": "error",   # 3.x 无 print_verbose，用 log_level 控制日志
  })
  ```
  不依赖包默认值，保证确定性，未来版本升级不漂移。
- `Global.log_level: "error"`，stdout 只输出结果，日志/错误走 stderr。

### 2.3 推理参数（官方默认即最佳；仅 text_score 一处有意覆盖，见下行）

| 参数 | 值 | 说明 |
|---|---|---|
| `text_score` | **0.0（实现层）** | ⚠️ rapidocr 内置 `filter_by_text_score` 会**过滤**低于阈值的行（源码实测）。为落实"低置信标记不丢弃"，ocr.py 将 `text_score` 设为 0.0 关闭内置过滤，自行以 0.5 为界打 `low_conf` 标记 |
| `box_thresh` | 0.5 | 检测框保留阈值 |
| `unclip_ratio` | 1.6 | 检测框外扩 |
| `limit_side_len` | 736（`limit_type=min`） | 小图自动放大最小边到 736px，小字截图免手动预处理 |
| `rec_batch_num` | 6 | 官方：batch 再大不提速反而可能变差（rec 内部按宽高比排序分批，批内同宽高比更高效） |
| `use_det/use_cls/use_rec` | true | 检测 + 方向分类 + 识别全开 |

### 2.4 生命周期与性能

1. 引擎单例：进程内只初始化一次 `RapidOCR`，多文件/多 PDF 页共用。
2. 启动预热：初始化后用**内置含文字小图**（如 PIL 绘制的 128px"OCR 测试"图）跑一次完整管线（det→cls→rec），不计时——1px 空白图 det 无框时 rec 不会执行，预热不完整。
3. 批量处理：多文件/多页一次调用完成，rec 裁剪自动走 batch（batch=6）。
4. 图像缩放统一策略：**依赖 rapidocr 内置预处理**（`use_preprocess_img: true` + `max_side_len: 2000`），不在 ocr.py 重复实现缩放；PDF 页渲染目标为**像素量 ~2MP 上限**（A4 约 150dpi），保证文本清晰且低于内置 2000px 上限。
5. 首次下载：`venv_setup.sh` 安装依赖后立即跑一次烟雾测试（同内置预热图）触发模型下载并校验，之后完全离线。

## 3. 总体架构

### 3.1 目录结构

```
ocr/                                # ~/.agents/skills/ocr（符号链接到本仓库）
├── SKILL.md                        # 触发条件 + 工作流 + 输出契约
├── scripts/
│   ├── venv_setup.sh               # uv 建 Python 3.12 venv + 装依赖 + 预热（幂等）
│   ├── ocr.py                      # 识别 CLI：图片/PDF → 文本/JSON/表格TSV
│   └── analyze.py                  # 审查 CLI：像素+坐标 → 审查事实清单(JSON)
├── references/
│   ├── output-format.md            # JSON 结构文档（按需加载）
│   └── layout-guide.md             # 布局/UI 推断指南（按需加载）
├── evals/
│   └── evals.json                  # 测试用例
└── docs/superpowers/specs/         # 本设计文档
```

### 3.2 组件职责

| 组件 | 职责 | 输入 → 输出 | 依赖 |
|---|---|---|---|
| `ocr.py` | PP-OCRv6 检测→方向→识别，阅读顺序排序 | 文件路径 → stdout 文本 / JSON / TSV | rapidocr, pymupdf, pillow, numpy |
| `analyze.py` | 像素采样 + bbox 坐标分析 → 审查事实（内部复用 rapidocr 拿文本块，自包含） | 图片路径 → stdout JSON | rapidocr, pillow, numpy |
| `venv_setup.sh` | 幂等初始化环境 + 模型预热 | 无 → 就绪的 .venv | uv |
| `SKILL.md` | 教 LLM 何时触发、怎么调脚本、怎么用结果 | — | — |
| references/* | 按需加载的格式文档与推断指南 | — | — |

### 3.3 环境

- Python 3.12.12（uv 管理，本机已缓存，规避 3.14 与 onnxruntime 的不兼容）
- 依赖：`rapidocr>=3.9`、`onnxruntime`（⚠️ rapidocr 3.9.2 无任何 extras，`[onnxruntime]` 写法不存在，需显式安装）、`pymupdf`、`pillow`、`numpy`
- venv 位于 `~/.agents/skills/ocr/.venv`（随技能迁移）

## 4. 数据流与输出格式

### 4.1 数据流

```
用户请求图片/PDF 内容
   │
   ▼
SKILL.md 判断意图
   ├─ 识别/理解/布局 → bash ocr.py <文件...> [--json|--table]
   ├─ 设计审查        → bash analyze.py <文件>（自包含：内部 OCR + 像素分析，一条命令）
   └─ 审查+布局都要   → analyze.py + ocr.py --json（先 analyze 出审查事实，需要布局再补 --json）
   │
   ▼
LLM 基于结构化结果 → 摘要/翻译/提取字段/表格化/布局描述/审查报告
```

### 4.2 ocr.py 输出契约

**模式一：纯文本（默认）**
```
[文件1/共3] 文件: report.pdf（页 1/共 10）
第1行文本...
第2行文本...                       ← 阅读顺序，空行分隔段落/块

[文件1/共3] 文件: report.pdf（页 2/共 10）
...
```
- 分隔头格式固定为 `[文件i/文件总数] 文件: <名>（页 p/总页数）`，文件序号与页码不混淆
- 阅读顺序：y 坐标聚类 → 行分组 → 行内按 x 排序 → 行间按 y 排序
- 低置信度文本保留，加 `⟦低置信⟧` 前缀标记
- 结果源：`RapidOCROutput`（`.boxes`/`.txts`/`.scores`/`.elapse`；3.9.x `__call__` 返回对象而非旧版 tuple，boxes 已映射回原图坐标）

**模式二：JSON（布局/UI 场景）**
```json
{
  "file": "screenshot.png",
  "page": 1,
  "width": 1280, "height": 800,
  "lines": [
    {"text": "商品详情", "conf": 0.98, "font_size": 36,
     "box": [12, 20, 108, 52], "low_conf": false}
  ]
}
```
- `box` = `[x1, y1, x2, y2]`，为检测多边形（四顶点）的**外接矩形**，原图像素坐标系
- `font_size` ≈ 框高（供字号层级推断）
- 编码契约：UTF-8，`ensure_ascii=False`（中文不转义），LLM 直接可读
- 多文件多页 → 顶层数组

**模式三：表格 TSV（`--table`）**
- 行聚类(y) × 列聚类(x) → TSV，空单元格留空，表头猜测由 LLM 做
- 列边界算法：对所有文本框的 x1 与 x2 做一维聚类 → 得列分隔线；单元格 = 行带 × 列带的交集（框跨多列的文本归入跨列单元，由 LLM 解读）
- 复杂表格（合并单元格/斜线）输出"原始 JSON 兜底"提示，LLM 从 bbox 自行推理

### 4.3 analyze.py 输出契约（审查事实清单，JSON）

**自包含设计**：analyze.py 内部复用 rapidocr 完成检测+识别拿到文本块（含 bbox/字号），再叠加像素分析，一条命令完成全部四项检查，不与 ocr.py 产生调用耦合（避免 LLM 为了审查要跑两次脚本）。

**输入约束**：analyze.py **仅接受图片文件**（不支持 PDF）；如用户要求对 PDF 页面做设计审查，由 SKILL.md 指导先渲染页面为图片再调用。

```json
{
  "file": "ui.png",
  "palette": [{"hex": "#1F2937", "pct": 42.1, "role": "bg"}, ...],
  "contrast_issues": [
    {"text": "保存", "box": [...], "fg": "#9CA3AF", "bg": "#F3F4F6",
     "ratio": 2.1, "wcag": "AA 未达标(需≥4.5)"}
  ],
  "font_size_clusters": [
    {"size": 36, "count": 3, "texts": ["商品详情", "加入购物车"], "consistent": true}
  ],
  "alignment_notes": [
    {"type": "left_align_group", "x": 120, "elements": 5}
  ]
}
```

四项检查实现要点：
1. **对比度**：⚠️ 文本块**中心像素不一定落在文字上**（空心字/描边字/间隙）。正确做法：对文本块内像素做颜色聚类分离"前景/背景"两簇（中心+环形采样互补，多采样点取众数）→ WCAG 公式算 ratio → 对照 4.5:1(AA)，只报事实不判断
2. **字号一致性**：font_size 聚类，孤立值标 `consistent: false`
3. **对齐一致性**：分别对 x1（左对齐）、x2（右对齐）、中心 x（居中对齐）做一维聚类，≥3 元素成组，离群元素标出
4. **配色统计**：降采样 k-means(≈5) 主色 + 占比；`role` 为启发式提示（占比最大者标 `"bg"`），仅供 LLM 参考，不作断言

### 4.4 错误处理

| 情况 | 行为 |
|---|---|
| 文件不存在/损坏 | stderr 报错，exit 1，LLM 向用户说明 |
| 无检测框 | 输出空结果 + "未检测到文字"提示 |
| analyze 遇无文字图 | 仅输出 palette（contrast_issues/alignment_notes 为空数组） |
| PDF 加密 | 报错提示需先解密 |
| 超大图 | 依赖 rapidocr 内置缩放（`use_preprocess_img` + `max_side_len=2000`），不自行降采样 |

## 5. SKILL.md 指令设计

### 5.1 结构

```
---
name: ocr
description: <完整触发词在实施阶段编写 SKILL.md 时产出，遵循 skill-creator 触发优化流程>
---

# OCR Skill

## 何时使用（触发）
## 快速上手（首次运行）
## 工作流（3 步：判断意图 → 调脚本 → 用结果）
## 脚本调用契约（3 种模式 + analyze.py，含示例命令）
## 阅读顺序与布局推断指南（如何从 JSON 坐标推层级/对齐/UI 结构）
## 表格处理指南（TSV 如何用、何时转原始 JSON）
## 设计审查指南（4 项检查的事实如何使用）
## 边界与诚实声明
```

### 5.2 关键设计决策（针对非多模态弱模型优化）

1. **指令下沉为可执行契约**：写"运行这个命令 → 得到这个格式 → 这样解读"，不写抽象原则。
2. **显式决策表**：用户意图 → 命令/模式，用表格列清楚。
3. **低置信度引导**：`⟦低置信⟧`/`low_conf` ≠ 错误，是让模型复核或向用户说明，不是删除。
4. **token 预算提示**：默认模式即纯文本。>3 张图或 >10 页 PDF 时**不要**全量 `--json`（token 过大）；确有布局/审查需求再对**单文件**补 `--json` 或 analyze.py。
5. **错误恢复**：脚本报错先看 stderr；venv 未建 → 先跑 setup，写死在 SKILL.md。

## 6. 验证方案（skill-creator 双跑流程）

| 阶段 | 内容 |
|---|---|
| 1. 素材 | 程序生成 + 真实采集：中文网页截图、聊天记录、英文界面、表格、多页 PDF、低对比度 UI 截图 |
| 2. 单元自测 | ocr.py 各模式跑通，人工核对准确性（对比度/字号/对齐/配色抽查） |
| 3. evals | `evals/evals.json` 6-8 个用例：提取发票字段、表格转 CSV、描述页面布局、UI 审查报告、PDF 摘要、翻译截图文字 |
| 4. 双跑 | 每用例并行跑"带 skill" vs "不带 skill"基线，量化断言（关键文本包含、TSV 单元格正确、JSON 合法、审查项检出） |
| 5. 评审 | eval-viewer 生成报告 → 用户评审 → 迭代 |

## 7. 风险与边界（诚实声明，写进 SKILL.md）

| 边界 | 说明 | 对策 |
|---|---|---|
| 输入形式 | 仅支持本地文件路径；URL 需先下载（由模型自行 curl 完成） | SKILL.md 写明 |
| 多栏布局（双栏报纸/多栏 dashboard） | y 聚类会把左右栏文本混排，阅读顺序不理想 | SKILL.md 提示：改用 `--json`，LLM 按 box.x 坐标自行分栏重排 |
| 旋转 90° 的图片 | cls 仅处理 0/180° 翻转，90° 旋转识别率下降 | SKILL.md 提示模型先用 PIL/sips 旋转回正再识别 |
| 图标/图形/颜色语义 | OCR 拿不到 | 布局描述只讲位置/层级/文字，不虚构视觉元素 |
| 手写体 | 准确率明显下降 | SKILL.md 声明；低置信标记兜底 |
| 复杂表格（合并单元格/斜线表头） | TSV 会错位 | 自动提示改用原始 JSON 由 LLM 兜底推理 |
| 多语混排（中英） | PP-OCRv6 多语模型可处理 | 默认 medium 覆盖 |
| analyze.py 颜色采样 | 文字与背景交界处可能采错 | 环形采样 + 多采样点取众数 |

## 8. 实施顺序（供 writing-plans 参考）

1. git init + 目录骨架（含 `.gitignore`：`.venv/`、`models/` 缓存、测试产物）
2. venv_setup.sh（uv venv + 依赖 + 预热下载）
3. ocr.py（引擎封装 → 预处理 → 识别 → 阅读顺序 → 三模式输出）
4. analyze.py（四检查项）
5. 测试素材生成 + 单元自测调优（含 tiny/small/medium 速度精度实测）
6. SKILL.md + references
7. evals 双跑 + eval-viewer 评审迭代
8. 符号链接安装到 ~/.agents/skills/ocr
