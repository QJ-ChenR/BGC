# BGC

基于 MIBiG 数据的生物合成基因簇项目。目前已实现核心生物合成基因提取；后续规划见 [DESIGN.md](DESIGN.md)。

## 环境配置

```bash
conda env create -f environment.yml
conda activate BGC
```

如果已经创建了 `BGC` 环境，只需激活。Biopython 在 Python 中的导入名称为 `Bio`。

## 输入数据

将 MIBiG 4.0 的 GenBank 文件放在 `data/raw/mibig_gbk_4.0/`，每个条目一个 `.gbk` 文件。
原始数据和生成结果放在 `data/` 下，不纳入 Git 版本控制。

## 提取核心基因

在项目根目录运行：

```bash
python script/extract_core_gene.py
```

也可以指定输入和输出目录：

```bash
python script/extract_core_gene.py --input-dir /path/to/gbk --output-dir /path/to/output
```

脚本仅选择 `CDS` feature 中 `gene_kind` 精确为 `biosynthetic` 的条目，跳过没有该标签的文件。
`biosynthetic-additional` 不纳入筛选；该筛选反映已有的 antiSMASH core 分类，不代表所有实验验证必需基因。

默认输出到 `data/processed/core_genes/`：

| 文件 | 内容 |
| --- | --- |
| `core_genes.faa` | CDS 注释中已有的蛋白翻译 |
| `core_genes.fna` | 按编码方向提取的 CDS 核酸序列，支持反向链和分段位置 |
| `core_genes.tsv` | BGC 编号、基因标识、坐标、功能证据及序列长度 |

FASTA 标识符格式为 `BGC编号|文件内CDS序号|基因标识`；CDS 序号从 1 开始，包含未入选的 CDS。
缺少蛋白翻译的核心 CDS 仅输出核酸和注释，并在运行摘要中计数。
TSV 的 `start/end` 为 1-based、两端包含的范围；`location_0based` 保留完整的 Biopython 位置表示。
重新运行会覆盖输出目录下的同名结果。

## 验证记录

本地 MIBiG 4.0 数据集共 2,636 个 GBK 文件：2,269 个文件含 core 标签，提取 6,831 个核心 CDS，跳过 367 个文件。
已核对三个输出文件的标识符和序列长度、`BGC0000001` 的 `abyB1/abyB2/abyB3`，以及反向链核酸序列。
