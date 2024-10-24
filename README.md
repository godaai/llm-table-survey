# LLM-Table-Survey
## Table of Contents

- [LLM-Table-Survey](#llm-table-survey)
  - [Table of Contents](#table-of-contents)
  - [📄 Paper List](#-paper-list)
    - [Large Language Model](#large-language-model)
    - [Pre-LLM Era Table Training](#pre-llm-era-table-training)
    - [Table Instruction-Tuning](#table-instruction-tuning)
    - [Code LLM](#code-llm)
    - [Hybrid of Table \& Code](#hybrid-of-table--code)
    - [Multimodal Table Understanding \& Extraction](#multimodal-table-understanding--extraction)
    - [Representation](#representation)
    - [Prompting](#prompting)
      - [NL2SQL](#nl2sql)
      - [Table QA](#table-qa)
      - [Spreadsheet](#spreadsheet)
      - [Multi-task Framework](#multi-task-framework)
    - [Tools](#tools)
    - [Survey](#survey)
  - [📊 Datasets \& Benchmarks](#-datasets--benchmarks)
    - [Benchmarks](#benchmarks)
    - [Datasets](#datasets)

## 📄 Paper List

### Large Language Model

* GPT-3, Language Models are Few-Shot Learners. NeurIPS 20. \[[Paper](https://papers.nips.cc/paper/2020/hash/1457c0d6bfcb4967418bfb8ac142f64a-Abstract.html)\]
* T5, Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer. \[[Paper](https://jmlr.org/papers/v21/20-074.html)\]
* FLAN, Finetuned Language Models Are Zero-Shot Learners. ICLR 22. \[[Paper](https://openreview.net/pdf?id=gEZrGCozdqR)\] \[[Code](https://github.com/google-research/FLAN)\]
* DPO, Direct Preference Optimization: Your Language Model is Secretly a Reward Model. NeurIPS 23. \[[Paper](https://arxiv.org/pdf/2305.18290)\]
* PEFT, The Power of Scale for Parameter-Efficient Prompt Tuning. EMNLP 21. \[[Paper](https://aclanthology.org/2021.emnlp-main.243.pdf)\]
* LoRA, LoRA: Low-rank Adaptation of Large Language Models. ICLR 22. \[[Paper](https://arxiv.org/pdf/2106.09685)\]
* Chain-of-thought Prompting, Chain-of-thought prompting elicits reasoning in large language models. NeurIPS 22. \[[Paper](https://proceedings.neurips.cc/paper_files/paper/2022/hash/9d5609613524ecf4f15af0f7b31abca4-Abstract-Conference.html)\]
* Least-to-most Prompting, Least-to-most prompting enables complex reasoning in large language models. ICLR 23. \[[Paper](https://openreview.net/pdf?id=WZH7099tgfM)\]
* Self-consistency Prompting,	Self-consistency improves chain of thought reasoning in language models. ICLR 23. \[[Paper](https://openreview.net/pdf?id=1PL1NIMMrw)\]
* ReAct, ReAct: Synergizing Reasoning and Acting in Language Models. ICLR 23. \[[Paper](https://openreview.net/forum?id=WE_vluYUL-X)\] \[[Code](https://github.com/ysymyth/ReAct)\]

### Pre-LLM Era Table Training

* TaBERT, TaBERT: Pretraining for Joint Understanding of Textual and Tabular Data. ACL 20 Main. \[[Paper](https://aclanthology.org/2020.acl-main.745/)\] \[[Code](https://github.com/facebookresearch/TaBERT)\]
* TaPEx, TAPEX: Table Pre-training via Learning a Neural SQL Executor. ICLR 22. \[[Paper](https://openreview.net/pdf?id=O50443AsCP)\] \[[Code](https://github.com/microsoft/Table-Pretraining)\] \[[Models](https://huggingface.co/models?search=microsoft/tapex)\]
* TABBIE, TABBIE: Pretrained Representations of Tabular Data. NAACL 21 Main. \[[Paper](https://aclanthology.org/2021.naacl-main.270/)\] \[[Code](https://github.com/SFIG611/tabbie)\]
* TURL, TURL: Table Understanding through Representation Learning. VLDB 21. \[[Paper](https://www.vldb.org/pvldb/vol14/p307-deng.pdf)\] \[[Code](https://github.com/sunlab-osu/TURL)\]
* RESDSQL, RESDSQL: Decoupling Schema Linking and Skeleton Parsing for Text-to-SQL. AAAI 23. \[[Paper](https://ojs.aaai.org/index.php/AAAI/article/view/26535/26307)\] \[[Code](https://github.com/RUCKBReasoning/RESDSQL)\]
* UnifiedSKG, UnifiedSKG: Unifying and Multi-Tasking Structured Knowledge Grounding with Text-to-Text Language Models. EMNLP 22 Main. \[[Paper](https://aclanthology.org/2022.emnlp-main.39/) \] \[[Code](https://github.com/xlang-ai/UnifiedSKG)\]
* SpreadsheetCoder, SpreadsheetCoder: Formula Prediction from Semi-structured Context. ICML 21. \[[Paper](https://arxiv.org/abs/2106.15339)\] \[[Code](https://github.com/google-research/google-research/tree/master/spreadsheet_coder)\]


### Table Instruction-Tuning

* Table-GPT, Table-GPT: Table-tuned GPT for Diverse Table Tasks. arXiv 2023. \[[Paper](https://arxiv.org/abs/2310.09263)\]
* TableLlama, TableLlama: Towards Open Large Generalist Models for Tables. NAACL 24. \[[Paper](https://arxiv.org/abs/2311.09206)\] \[[Code](https://github.com/OSU-NLP-Group/TableLlama)\] \[[Model: TableLlama 7B](https://huggingface.co/osunlp/TableLlama)\] \[[Dataset: TableInstruct](https://huggingface.co/datasets/osunlp/TableInstruct)\]

### Code LLM

* Codex, Evaluating Large Language Models Trained on Code. arXiv 21. \[[Paper](https://arxiv.org/abs/2107.03374)\]
* StarCoder, StarCoder: may the source be with you!. TMLR 23. \[[Paper](https://arxiv.org/abs/2305.06161)\] \[[Code](https://github.com/bigcode-project/starcoder)\] \[[Models](https://huggingface.co/bigcode/starcoder)\]
* Code Llama, Code Llama: Open Foundation Models for Code. arXiv 23. \[[Paper](https://arxiv.org/abs/2308.12950)\] \[[Code](https://github.com/meta-llama/codellama)\]
* WizardLM, WizardLM: Empowering Large Pre-Trained Language Models to Follow Complex Instructions. ICLR 24. \[[Paper](https://openreview.net/forum?id=CfXh93NDgH)\] \[[Model: WizardLM 13B](https://huggingface.co/WizardLMTeam/WizardLM-13B-V1.0)\] \[[Model: WizardLM 70B](https://huggingface.co/WizardLMTeam/WizardLM-70B-V1.0)\]
* WizardCoder, WizardCoder: Empowering Code Large Language Models with Evol-Instruct. ICLR 24. \[[Paper](https://openreview.net/forum?id=UnUwSIgK5W)\] \[[Code](https://github.com/nlpxucan/WizardLM)\] \[[Models: WizardCoder 15B](https://huggingface.co/WizardLMTeam/WizardCoder-15B-V1.0)\]
* Magicoder, Magicoder: Source Code Is All You Need. ICML 24. \[[Paper](https://arxiv.org/abs/2312.02120)\] \[[Code](https://github.com/ise-uiuc/magicoder)\] \[[Models 6.7B/7B](https://huggingface.co/models?search=ise-uiuc/Magicoder)\]
* Lemur, Lemur: Harmonizing Natural Language and Code for Language Agents. ICLR 24. \[[Paper](https://openreview.net/forum?id=hNhwSmtXRh)\] \[[Code](https://github.com/OpenLemur/Lemur)\] \[[Model: Lemur 70B](https://huggingface.co/OpenLemur/lemur-70b-v1)\] \[[Model: Lemur 70B Chat](https://huggingface.co/OpenLemur/lemur-70b-chat-v11)\]
* InfiAgent-DABench, InfiAgent-DABench: Evaluating Agents on Data Analysis Tasks. ICML 24. \[[Paper](https://arxiv.org/abs/2401.05507)\] \[[Code](https://github.com/InfiAgent/InfiAgent)\]

### Hybrid of Table & Code
* TableLLM, TableLLM: Enabling Tabular Data Manipulation by LLMs in Real Office Usage Scenarios. \[[Paper](http://arxiv.org/abs/2403.19318)\] \[[Model TableLLM 7B](https://huggingface.co/RUCKBReasoning/TableLLM-7b)\] \[[Model TableLLM 13B](https://huggingface.co/RUCKBReasoning/TableLLM-13b)\]
* StructLM, StructLM: Towards Building Generalist Models for Structured Knowledge Grounding. arXiv 24. \[[Paper](https://arxiv.org/abs/2402.16671)\] \[[Model: StructLM 7B](https://huggingface.co/TIGER-Lab/StructLM-7B)\] \[[Model: StructLM 13B](https://huggingface.co/TIGER-Lab/StructLM-13B)\] \[[Model: StructLM 34B](https://huggingface.co/TIGER-Lab/StructLM-34B)\] \[[Dataset: SKGInstruct](https://huggingface.co/datasets/TIGER-Lab/SKGInstruct)\]

### Parameter-Efficient Fine-Tuning

* FinSQL, FinSQL: Model-Agnostic LLMs-based Text-to-SQL Framework for Financial Analysis. SIGMOD Companion 24. [\[Paper\](https://arxiv.org/pdf/2401.10506)]

### Direct Preference Optimization

* SENSE, Synthesizing Text-to-SQL Data from Weak and Strong LLMs. ACL 24. \[[Paper](https://aclanthology.org/2024.acl-long.425.pdf)\]


### Small Language Model + Large Language Model

* ZeroNL2SQL, Combining Small Language Models and Large Language Models for Zero-Shot NL2SQL. VLDB 24. \[[Paper](https://dl.acm.org/doi/10.14778/3681954.3681960)\]

### Multimodal Table Understanding & Extraction

* LayoutLM, LayoutLM: Pre-training of Text and Layout for Document Image Understanding. KDD 20. \[[Paper](https://dl.acm.org/doi/10.1145/3394486.3403172)\]
* PubTabNet, Image-Based Table Recognition: Data, Model, and Evaluation. ECCV 20. \[[Paper](https://link.springer.com/chapter/10.1007/978-3-030-58589-1_34)\] \[[Code & Data](https://github.com/ibm-aur-nlp/PubTabNet)\]
* Table-LLaVA, Multimodal Table Understanding. ACL 24. \[[Paper](https://arxiv.org/abs/2406.08100)\] \[[Code](https://github.com/SpursGoZmy/Table-LLaVA)\] \[[Model](https://huggingface.co/SpursgoZmy/table-llava-v1.5-7b)\]
* TableLVM, TableVLM: Multi-modal Pre-training for Table Structure Recognition. ACL 23. \[[Paper](https://aclanthology.org/2023.acl-long.137/)\]
* PixT3, PixT3: Pixel-based Table-To-Text Generation. ACL 24. \[[Paper](https://aclanthology.org/2024.acl-long.364.pdf)\]

### Representation

* Tabular representation, noisy operators, and impacts on table structure understanding tasks in LLMs. NeurIPS 2023 second table representation learning workshop. \[[Paper](https://openreview.net/forum?id=Ld5UCpiT07)\]
* SpreadsheetLLM, SpreadsheetLLM: Encoding Spreadsheets for Large Language Models. arXiv 24. \[[Paper](http://arxiv.org/abs/2407.09025)\]
* Enhancing Text-to-SQL Capabilities of Large Language Models: A Study on Prompt Design Strategies. EMNLP 23.  \[[Paper](https://aclanthology.org/2023.findings-emnlp.996.pdf)\] \[[Code](https://github.com/linyongnan/STRIKE)\]
* Tables as Texts or Images: Evaluating the Table Reasoning Ability of LLMs and MLLMs. arXiv 24. \[[Paper](http://arxiv.org/abs/2402.12424)\]

### Prompting
#### NL2SQL

* The Dawn of Natural Language to SQL: Are We Fully Ready? VLDB 24. \[[Paper](http://arxiv.org/abs/2406.01265)\] \[[Code](https://github.com/HKUSTDial/NL2SQL360)\]
* MCS-SQL, MCS-SQL: Leveraging Multiple Prompts and Multiple-Choice Selection For Text-to-SQL Generation. \[[Paper](http://arxiv.org/abs/2405.07467)\]
* DIN-SQL, DIN-SQL: Decomposed In-Context Learning of Text-to-SQL with Self-Correction	Prompting, Decompose. NeurIPS 23. \[[Paper](https://proceedings.neurips.cc/paper_files/paper/2023/hash/72223cc66f63ca1aa59edaec1b3670e6-Abstract-Conference.html)\] \[[Code](https://github.com/MohammadrezaPourreza/Few-shot-NL2SQL-with-prompting)\]
* DAIL-SQL, Text-to-SQL Empowered by Large Language Models: A Benchmark Evaluation. VLDB 24. \[[Paper](https://arxiv.org/abs/2308.15363)\] \[[Code](https://github.com/BeachWang/DAIL-SQL)\]
* C3, C3: Zero-shot Text-to-SQL with ChatGPT. arXiv 24. \[[Paper](https://arxiv.org/abs/2307.07306)\] \[[Code](https://github.com/bigbigwatermalon/C3SQL)\]

#### Table QA

* Dater, Large Language Models are Versatile Decomposers: Decompose Evidence and Questions for Table-based Reasoning. SIGIR 23. \[[Paper](https://arxiv.org/abs/2301.13808)\] \[[Code](https://github.com/AlibabaResearch/DAMO-ConvAI)\]
* Binder, Binding language models in symbolic languages. ICLR 23. \[[Paper](https://arxiv.org/abs/2210.02875)\] \[[Code](https://github.com/xlang-ai/Binder)\]
* ReAcTable, ReAcTable: Enhancing ReAct for Table Question Answering. VLDB 24. \[[Paper](https://arxiv.org/abs/2310.00815)\] \[[Code](https://github.com/yunjiazhang/ReAcTable)\]
* E5, E5: Zero-shot Hierarchical Table Analysis using Augmented LLMs via Explain, Extract, Execute, Exhibit and Extrapolate. NAACL 24. \[[Paper](https://aclanthology.org/2024.naacl-long.68/)\] \[[Code](https://github.com/zzh-SJTU/E5-Hierarchical-Table-Analysis)\]
* Chain-of-Table, Chain-of-Table: Evolving Tables in the Reasoning Chain for Table Understanding. ICLR 24. \[[Paper](https://arxiv.org/abs/2401.04398)\]
* ITR, An Inner Table Retriever for Robust Table Question Answering. ACL 23. \[[Paper](https://aclanthology.org/2023.acl-long.551)\]
* LI-RAGE, LI-RAGE: Late Interaction Retrieval Augmented Generation with Explicit Signals for Open-Domain Table Question Answering. ACL 23. \[[Paper](https://aclanthology.org/2023.acl-short.133)\]

#### Spreadsheet

* SheetCopilot, SheetCopilot: Bringing Software Productivity to the Next Level through Large Language Models Agent. NeurIPS 23. \[[Paper](https://arxiv.org/abs/2305.19308)\] \[[Code](https://sheetcopilot.github.io/)\]
* SheetAgent, SheetAgent: A Generalist Agent for Spreadsheet Reasoning and Manipulation via Large Language Models. arXiv 24. \[[Paper](http://arxiv.org/abs/2403.03636)\]
* Vision Language Models for Spreadsheet Understanding: Challenges and Opportunities. arXiv 24. \[[Paper](http://arxiv.org/abs/2405.16234)\]

#### Multi-task Framework

* StructGPT, StructGPT: A General Framework for Large Language Model to Reason over Structured Data. EMNLP 23 Main. \[[Paper](https://aclanthology.org/2023.emnlp-main.574/)\] \[[Code](https://github.com/RUCAIBox/StructGPT)\]
* TAP4LLM, TAP4LLM: Table Provider on Sampling, Augmenting, and Packing Semi-structured Data for Large Language Model Reasoning. arXiv 23. \[[Paper](https://arxiv.org/abs/2312.09039)\]
* UniDM, UniDM: A Unified Framework for Data Manipulation with Large Language Models. MLSys 24. \[[Paper](https://arxiv.org/abs/2405.06510)\]
* Data-Copilot, Data-Copilot: Bridging Billions of Data and Humans with Autonomous Workflow. arXiv 23. \[[Paper](https://arxiv.org/abs/2306.07209)\] \[[Code](https://github.com/zwq2018/Data-Copilot)\]

### Tools

* [LlamaIndex](https://github.com/run-llama/llama_index)
* [PandasAI](https://github.com/sinaptik-ai/pandas-ai)
* [Vanna](https://github.com/vanna-ai/vanna)
* DB-GPT. DB-GPT: Empowering Database Interactions with Private Large Language Models. \[[Paper](http://arxiv.org/abs/2312.17449)\] \[[Code](https://github.com/eosphoros-ai/DB-GPT)\]
* RetClean. RetClean: Retrieval-Based Data Cleaning Using Foundation Models and Data Lakes. \[[Paper](http://arxiv.org/abs/2303.16909)\] \[[Code](https://github.com/qcri/RetClean)\]


### Survey

* A Survey of Large Language Models. \[[Paper](http://arxiv.org/abs/2303.18223)\]
* A Survey on Large Language Model Based Autonomous Agents. \[[Paper](https://link.springer.com/10.1007/s11704-024-40231-1)\]
* Table Pre-training: A Survey on Model Architectures, Pre-training Objectives, and Downstream Tasks. \[[Paper](https://www.ijcai.org/proceedings/2022/761)\]
* Transformers for tabular data representation: A survey of models and applications. \[[Paper](https://aclanthology.org/2023.tacl-1.14)\]
* A Survey of Table Reasoning with
Large Language Models. \[[Paper](http://arxiv.org/abs/2402.08259)\]
* A survey on table question answering: Recent advances. \[[Paper](https://doi.org/10.1007/978-981-19-7596-7_14)\]
* Large Language Models(LLMs) on Tabular Data - A Survey. \[[Paper](http://arxiv.org/abs/2402.17944)\]
* A Survey on Text-to-SQL Parsing: Concepts, Methods, and Future Directions. \[[Paper](http://arxiv.org/abs/2208.13629)\]

## 📊 Datasets & Benchmarks

### Benchmarks

| Name               | Keywords                  | Artifact                                                                 | Paper                                                       |
|--------------------|---------------------------|--------------------------------------------------------------------------|-------------------------------------------------------------|
| MBPP               | Code    | [link](https://huggingface.co/datasets/mbpp)                             | [arXiv 21](https://arxiv.org/abs/2108.07732)                |
| HumanEval          | Code    | [link](https://github.com/openai/human-eval)                         | [arXiv 21](https://arxiv.org/abs/2107.03374)                |
| Dr.Spider         | NL2SQL, Robustness               | [link](https://github.com/awslabs/diagnostic-robustness-text-to-sql) | [ICLR 23](https://arxiv.org/abs/2301.08881)                 |
| WiKiTableQuestions | Table QA                  | [link](https://github.com/ppasupat/WikiTableQuestions)               | [ACL 15](https://aclanthology.org/P15-1142/)                  |
| WiKiSQL            | Table QA,NL2SQL      | [link](https://github.com/salesforce/WikiSQL)                        | [arXiv 17](https://arxiv.org/abs/1709.00103)                                                |
| TabFact            | Table Fact Verification                  | [link](https://tabfact.github.io/)                                   | [ICLR 20](https://arxiv.org/abs/1909.02164)                 |
| HyBirdQA           | Table QA                  | [link](https://github.com/wenhuchen/HybridQA)                        | [EMNLP 20](https://arxiv.org/abs/2004.07347)                |
| FetaQA             | Table Fact Verification                  | [link](https://github.com/Yale-LILY/FeTaQA)                          | [TACL 22](https://aclanthology.org/2022.tacl-1.3/)                 |
| RobuT              | Table QA                  | [link](https://github.com/yilunzhao/RobuT)                           | [ACL 23](https://arxiv.org/abs/2306.14321)                  |
| AnaMeta            | Table Metadata            | [link](https://github.com/microsoft/AnaMeta)                         | [ACL 23](https://arxiv.org/abs/2209.00946)                  |
| GPT4Table          | Table QA,   Table-to-text | [link](https://github.com/Y-Sui/GPT4Table)                           | [WSDM 24](https://arxiv.org/abs/2305.13062)                 |
| ToTTo              | Table-to-text             | [link](https://github.com/google-research-datasets/totto)            | [EMNLP 20](https://aclanthology.org/2020.emnlp-main.89/)                |
| SpreadsheetBench | Spreadsheet Manipulation | [link](https://github.com/RUCKBReasoning/SpreadsheetBench) | [NeurIPS 24](https://arxiv.org/abs/2406.14991) |
| BIRD               | NL2SQL               | [link](https://bird-bench.github.io/)                                | [NeurIPS 23](https://arxiv.org/abs/2305.03111)              |
| Spider             | NL2SQL               | [link](https://yale-lily.github.io/spider)                           | [EMNLP 18](https://arxiv.org/abs/1809.08887)                |
| Dr.Spider             | NL2SQL               | [link](https://github.com/awslabs/diagnostic-robustness-text-to-sql)                           | [ICLR 23](https://arxiv.org/abs/2301.08881)                |
| ScienceBenchmark             | NL2SQL               | [link](https://sciencebenchmark.cloudlab.zhaw.ch/)                           | [VLDB 24](https://arxiv.org/pdf/2306.04743)                |
| DS-1000            | Data Analysis    | [link](https://ds1000-code-gen.github.io/)                           | [ICML 23](https://arxiv.org/abs/2211.11501)                 |
| InfiAgent-DABench | Data Analysis | [link](https://github.com/InfiAgent/InfiAgent) | [ICML 24](https://arxiv.org/abs/2401.05507) |
| TableBank | Table Detection | [link](https://doc-analysis.github.io/tablebank-page/) | [LERC 20](https://aclanthology.org/2020.lrec-1.236/) | 
| PubTabNet | Table Extraction | [link](https://github.com/ibm-aur-nlp/PubTabNet) | [ECCV 20](https://arxiv.org/abs/1911.10683) |
| ComTQA | Visual Table QA, Table Detection, Table Extraction | [link](https://huggingface.co/datasets/ByteDance/ComTQA) | [arXiv 24](https://arxiv.org/abs/2406.01326v1) |


### Datasets


| Name               | Keywords                  | Artifact                                                                 | Paper                                                       |
|--------------------|---------------------------|--------------------------------------------------------------------------|-------------------------------------------------------------|
| TableInstruct                           | Table Instruction Tuning | [link](https://huggingface.co/datasets/osunlp/TableInstruct)                                          | [arXiv 23](https://arxiv.org/pdf/2311.09206.pdf)                                                           |
| WDC                | Web Table                 | [link](https://webdatacommons.org/)                                  | [WWW 16](https://dl.acm.org/doi/10.1145/2872518.2889386)    |
| GitTables          | GitHub CSVs    | [link](https://gittables.github.io/)                                 | [SIGMOD 23](https://arxiv.org/abs/2106.07258)              |
| DART               | Table-to-text             | [link](https://github.com/Yale-LILY/dart)                            | [NAACL 21](https://aclanthology.org/2021.naacl-main.37/)                |
| MMTab               | Multimodal Table Understanding             | [link](https://huggingface.co/datasets/SpursgoZmy/MMTab)                            | [ACL 24](https://arxiv.org/abs/2406.08100)                |
| SchemaPile | Database Schemas | [link](https://schemapile.github.io/) | [SIGMOD 24](https://dl.acm.org/doi/abs/10.1145/3654975) |