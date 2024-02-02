# LLM-Table-Survey
## Table of Contents

- [LLM-Table-Survey](#llm-table-survey)
  - [Table of Contents](#table-of-contents)
  - [Methods](#methods)
  - [Dataset \& Benchmark](#dataset--benchmark)

## Methods


| Name                                  | Keywords  | Artifact                                                                                 | Paper                                                                                                   |
|---------------------------------------|-----------|------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------|
| Codex                                 | Code Generation  | -                                                                                        | [arXiv 21](https://arxiv.org/abs/2107.03374)                                                            |
| GPT-3 | LLM, in-context learning    | -                                                                                        | [NeurIPS 20](https://proceedings.neurips.cc/paper/2020/file/1457c0d6bfcb4967418bfb8ac142f64a-Paper.pdf) |
| Spreadsheetcoder                      | Training  | [Code](https://github.com/google-research/google-research/tree/master/spreadsheet_coder) | [ICML 21](https://openreview.net/pdf?id=lH1PV42cbF)                                                             |
| Binder                                | Prompt    | [Code](https://github.com/xlang-ai/Binder)                                               | [ICLR 23](https://arxiv.org/abs/2210.02875)                                                             |
| TURL                                  | Table Pre-training  | [Code](https://github.com/sunlab-osu/TURL)                                               | [VLDB 20](https://www.vldb.org/pvldb/vol14/p307-deng.pdf)                                                             |
| BERT                                  | Pre-training  | [Model](https://huggingface.co/docs/transformers/model_doc/bert)                         | [NAACL 19](https://arxiv.org/abs/1810.04805)                                                            |
| C3                                    | Prompting    | [Code](https://arxiv.org/abs/2307.07306)                                                 | [arXiv 23](https://arxiv.org/abs/2307.07306)                                                            |
| DAIL-SQL                              | Prompting  | [Code](https://github.com/taoyds/test-suite-sql-eval)                                    | [arXiv 23](https://arxiv.org/abs/2308.15363)                                                            |
| TaPas                                 | Table Pre-training  | [Code](https://github.com/google-research/tapas)                                         | [ACL 20](https://aclanthology.org/2020.acl-main.398)                                                              |
| Tabbie                                | Table Pre-training  | [Code](https://github.com/SFIG611/tabbie)                                                | [ACL 21](https://aclanthology.org/2021.naacl-main.270)                                                            |
| StructGPT                             | Prompting, Structured Knowledge    | [Code](https://github.com/RUCAIBox/StructGPT)                                            | [EMNLP 23](https://aclanthology.org/2023.emnlp-main.574/)                                                            |
| DPR                                   | Retrieval-Augmented Generation  | [Code](https://github.com/facebookresearch/DPR)                                          | [EMNLP 20](https://arxiv.org/abs/2004.04906)                                                            |
| RESDSQL                               | fine-tuning  | [Code](https://github.com/RUCKBReasoning/RESDSQL)                                        | [AAAI 23](https://ojs.aaai.org/index.php/AAAI/article/view/26535/26307)                                                             |
| SheetCopilot                          | Agent, Spreadsheet    | [Code](https://sheetcopilot.github.io/)                                                  | [NeurIPS 23](https://arxiv.org/abs/2305.19308)                                                          |
| Table-GPT                              | Instruction-tuning  | -                                                                                        | [arXiv 23](https://arxiv.org/abs/2310.09263v1)                                                          |
| LI-RAGE                               | Training  | [Code](https://github.com/amazon-science/robust-tableqa)                                 | [ACL 23](https://aclanthology.org/2023.acl-short.133/)                                                  |
| TaPEx                                 | Table Pre-training  | [Code](https://github.com/microsoft/Table-Pretraining)                                   | [ICLR 22](https://openreview.net/pdf?id=O50443AsCP)                                                             |
| Prompt Design Strategies              | Prompt    | [Code](https://github.com/linyongnan/STRIKE)                                             | [EMNLP 23](https://aclanthology.org/2023.findings-emnlp.996.pdf)                                                            |
| DIN-SQL                               | Prompting, Decompose    | [Code](https://github.com/MohammadrezaPourreza/Few-shot-NL2SQL-with-prompting)           | [NeurIPS 23](https://arxiv.org/abs/2304.11015)                                                          |
| FLAN                                  | Instruction-tuning  | [Code](https://github.com/google-research/FLAN)                                          | [ICLR 22](https://openreview.net/pdf?id=gEZrGCozdqR)                                                             |
| UnifiedSKG                            | Instruction-tuning  | [Code](https://github.com/xlang-ai/UnifiedSKG)                                           | [EMNLP 22](https://aclanthology.org/2022.emnlp-main.39/)                                                            |
| DB-GPT                                | Industry Framework  | [Code](https://github.com/eosphoros-ai/DB-GPT)                                           | [arXiv 23](https://arxiv.org/abs/2312.17449)                                                            |
| ReAct                                 | Prompting    | [Code](https://react-lm.github.io/)                                                      | [ICLR 23](https://arxiv.org/abs/2210.03629)                                                             |
| Dater                                 | Prompting, Decomposing    | [Code](https://github.com/AlibabaResearch/DAMO-ConvAI)                                   | [SIGIR 23](https://arxiv.org/abs/2301.13808)                                                            |
| TaBERT                                | Table Pre-training  | [Code](http://fburl.com/TaBERT)                                                          | [ACL 20](https://aclanthology.org/2020.acl-main.745/)                                                              |
| TableLlama                            | Instruction-tuning  | [Model](https://huggingface.co/osunlp/TableLlama)                                        | [arXiv 23](https://arxiv.org/abs/2311.09206)                                                            |
| Data-Copilot                          | Agent, Data Visualization    | [Code](https://github.com/zwq2018/Data-Copilot)                                          | [arXiv 23](https://arxiv.org/abs/2306.07209)                                                            |
| ReAcTable                             | Agent, ReAct    | [Code](https://github.com/yunjiazhang/ReAcTable.git)                                     | [arXiv 23](https://arxiv.org/abs/2310.00815)                                                            |
| ITR                                   | Retrieval  | [Code](https://github.com/amazon-science/robust-tableqa)                                 | [ACL 23](https://aclanthology.org/2023.acl-long.551/)                                                                                                     |

## Dataset & Benchmark


| Name               | Keywords                  | Artifact                                                                 | Paper                                                       |
|--------------------|---------------------------|--------------------------------------------------------------------------|-------------------------------------------------------------|
| MBPP               | Code    | [link](https://huggingface.co/datasets/mbpp)                             | [arXiv 21](https://arxiv.org/abs/2108.07732)                |
| HumanEval          | Code    | [link](https://github.com/openai/human-eval)                         | [arXiv 21](https://arxiv.org/abs/2107.03374)                |
| Dr.Spider         | Text-to-SQL, Robustness               | [link](https://github.com/awslabs/diagnostic-robustness-text-to-sql) | [ICLR 23](https://arxiv.org/abs/2301.08881)                 |
| TabFact            | Table QA                  | [link](https://tabfact.github.io/)                                   | [ICLR 20](https://arxiv.org/abs/1909.02164)                 |
| HyBirdQA           | Table QA                  | [link](https://github.com/wenhuchen/HybridQA)                        | [EMNLP 20](https://arxiv.org/abs/2004.07347)                |
| AnaMeta            | Table Metadata            | [link](https://github.com/microsoft/AnaMeta)                         | [ACL 23](https://arxiv.org/abs/2209.00946)                  |
| InfiAgent-DABench  | Data Analysis    | [link](https://arxiv.org/abs/2401.05507)                             | [arXiv 24](https://arxiv.org/abs/2401.05507)                |
| GitTables          | GitHub CSVs    | [link](https://gittables.github.io/)                                 | [SIGMOD 23](https://arxiv.org/abs/2106.07258)              |
| DS-1000            | Data Analysis    | [link](https://ds1000-code-gen.github.io/)                           | [ICML 23](https://arxiv.org/abs/2211.11501)                 |
| WDC                | Web Table                 | [link](https://webdatacommons.org/)                                  | [WWW 16](https://dl.acm.org/doi/10.1145/2872518.2889386)    |
| BIRD               | Text-to-SQL               | [link](https://bird-bench.github.io/)                                | [NeurIPS 23](https://arxiv.org/abs/2305.03111)              |
| DART               | Table-to-text             | [link](https://github.com/Yale-LILY/dart)                            | [NAACL 21](https://aclanthology.org/2021.naacl-main.37/)                |
| FetaQA             | Table QA                  | [link](https://github.com/Yale-LILY/FeTaQA)                          | [TACL 22](https://aclanthology.org/2022.tacl-1.3/)                 |
| ToTTo              | Table-to-text             | [link](https://github.com/google-research-datasets/totto)            | [EMNLP 20](https://aclanthology.org/2020.emnlp-main.89/)                |
| WiKiTableQuestions | Table QA                  | [link](https://github.com/ppasupat/WikiTableQuestions)               | [ACL 15](https://aclanthology.org/P15-1142/)                  |
| sheetperf          | Spreadsheet              | [link](https://github.com/dataspread/spreadsheet-benchmark)          | [SIGMOD 20](https://dl.acm.org/doi/10.1145/3318464.3389782) |
| GPT4Table          | Table QA,   Table-to-text | [link](https://github.com/Y-Sui/GPT4Table)                           | [WSDM 24](https://arxiv.org/abs/2305.13062)                 |
| Spider             | Text-to-SQL               | [link](https://yale-lily.github.io/spider)                           | [EMNLP 18](https://arxiv.org/abs/1809.08887)                |
| RobuT              | Table QA                  | [link](https://github.com/yilunzhao/RobuT)                           | [ACL 23](https://arxiv.org/abs/2306.14321)                  |
| WiKiSQL            | Table QA,Text-to-SQL      | [link](https://github.com/salesforce/WikiSQL)                        | [arXiv 17](https://arxiv.org/abs/1709.00103)                                                |
| Observatory                           | Table Embedding Benchmark | [Code](https://github.com/superctj/observatory)                                          | [VLDB 24](https://arxiv.org/abs/2310.07736v3)                                                           |
