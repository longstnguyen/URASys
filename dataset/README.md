# QA Dataset Collection for Benchmarking Answerability and Ambiguity

We present a curated suite of Question Answering (QA) datasets specifically designed for evaluating model performance under challenging conditions: distinguishing answerable versus unanswerable queries, handling underspecified or ambiguous inputs, and performing multi-hop reasoning. The collection spans both English and Vietnamese resources and is standardized for reproducible benchmarking across systems.

To ensure consistency, each dataset is distributed in a unified CSV format with two main variants:

- **Standard QA sets (`*_1000.csv`)**: Balanced subsets for general answerability evaluation.
- **Ambiguity-focused subsets (`*_ambious.csv`)**: Specifically curated to test model behavior on underspecified or multi-interpretation queries.

## Dataset Descriptions

- **SQuAD 2.0** [1]  
SQuAD 2.0 is an English extractive QA dataset combining over 100,000 answerable and 50,000 unanswerable questions. It extends the original Stanford Question Answering Dataset by introducing adversarially crafted unanswerable queries that are contextually relevant but lack an answer in the passage. This design tests a model’s ability to balance span extraction with robust answerability detection, making SQuAD 2.0 a primary benchmark for English QA research.
  
- **UIT-ViQuAD 2.0** [2]  
UIT-ViQuAD 2.0 is a Vietnamese adaptation of SQuAD 2.0, developed for the VLSP 2021 Challenge. It contains over 23,000 questions aligned with Vietnamese syntax and linguistic characteristics, including both answerable and unanswerable cases. The dataset was created to advance Vietnamese NLP research and evaluate QA models’ ability to handle language-specific challenges such as tonal ambiguity and word segmentation.

- **HotpotQA** [3]  
HotpotQA is a large-scale English multi-hop extractive QA dataset with more than 113,000 question–answer pairs. Each question is designed to require reasoning over multiple passages and is annotated with supporting facts that enable evidence-based explanations. HotpotQA is widely used for training and benchmarking explainable QA systems and supports both a distractor setting with curated passages and a full-wiki retrieval setting.

- **VIMQA** [4]  
VIMQA is a Vietnamese multi-hop QA dataset introduced at LREC 2022 with over 15,000 question–answer pairs. It focuses on explainable reasoning and document-level context linking in Vietnamese, providing supporting facts for each answer to encourage transparency. VIMQA addresses the lack of multi-hop resources for low-resource languages and serves as a benchmark for explainable QA in Vietnamese.

  *For public datasets, we sample 1,000 test-set examples per dataset. For SQuAD 2.0 and UIT-ViQuAD 2.0, the sampling process guarantees inclusion of the unanswerable subsets.*

- **UniQA**  
UniQA is a custom dataset of real-world Vietnamese student queries on university admissions. Each question links to official academic documents, reflecting URASys’s target deployment scenario. Only a 2-sample preview is available during peer review; the full release follows paper acceptance.

## Ambiguity Subset Construction

The `*_ambious.csv` files are not direct subsets but are generated to probe model performance on underspecified inputs. The pipeline:

**1.** **Source Selection:** Start from the base QA dataset (e.g., SQuAD 2.0, ViQuAD 2.0).  

**2.** **Ambiguity Generation:**  
   - Use a language model to mask or alter the original question according to three ambiguity types:
     - `missing_context`: Remove key constraints (time, entity, scope).
     - `multiple_interpretations`: Create questions interpretable in multiple valid ways.
     - `generalization`: Overly broaden the query beyond the source context.
       
**3.** **Validation:**  
   - Automatically verify that the new question cannot be fully answered using the paragraph alone.
   - Annotate critical `info` fields required to disambiguate.
     
**4.** **Manual Spot-Check:** Random samples are human-reviewed for quality assurance.

Each ambiguous record contains:
- `question`: The underspecified or ambiguous version of the query.
- `answer`: The correct answer to the original question.
- `paragraph`: The source context passage used for answering or verifying the question.
- `info`: A JSON list of missing information the model should request to clarify.


## Dataset Variants and Format

| Variant         | File Pattern           | Fields                                                       |
|-----------------|------------------------|--------------------------------------------------------------|
| Standard (answerable and unanswerable)        | `*_1000.csv`           | `question`, `answer`, `paragraph`                           |
| Ambiguous       | `*_ambious.csv`        | `question`, `answer`, `paragraph`, `info`                   |
| UniQA (hidden)  | `UniQA_*.csv`          | `question`, `answer`, `paragraph`, `info`; only 2 rows visible |

> For UniQA, all rows beyond the first two are anonymized with a placeholder message.

## Dataset Statistics

| Dataset            | Answerable | Unanswerable | Ambiguous |
|--------------------|------------|--------------|-----------|
| SQuAD 2.0          | 508        | 492          | 418       |
| UIT-ViQuAD 2.0     | 800        | 200          | 731       |
| HotpotQA           | 1000       | 0            | 0         |
| VIMQA              | 1000       | 0            | 0         |
| UniQA              | 500        | 500          | 314       |

## File List

- `Hotpot_1000.csv`
- `Squad2_1000.csv`
- `Squad2_ambious.csv`
- `Viquad2_1000.csv`
- `Viquad2_ambious.csv`
- `VimQA_1000.csv`
- `UniQA_1000_anonymous.csv`
- `UniQA_ambious_anonymous.csv`

## Field Definitions

- `question`: The query posed to the QA system.
- `answer`: Ground truth answer string.
- `paragraph`: Supporting or non-supporting context.
- `info`: *(Ambiguous sets only)* Required disambiguation cues as a JSON list.

## JSON Examples

**Answerable:** direct, well-specified question.
```json
{
  "question": "Beyonce did an interview with which magazine and was asked about feminism?",
  "answer": "Vogue",
  "paragraph": "In an interview published by Vogue in April 2013, Beyoncé was asked if she considers herself a feminist..."
}
```

**Unanswerable:** questions with multiple interpretations, insufficient context, or overly broad scope.
```json
{
  "question": "Which Nigerian author did Beyonce collaborate with for a TEDx speech sample?",
  "answer": "No answer",
  "paragraph": "In an interview published by Vogue in April 2013, Beyoncé was asked if she considers herself a feminist..."
}
```

**Ambiguous:** specific query with no matching information in the database `info` contains a list of clarifying details the model should request.
```json
{
  "question": "Which campaign promoting female empowerment did she support?",
  "info": ["Which artist is being referred to", "Whether the campaign is music-related or social"],
  "answer": "Ban Bossy campaign",
  "paragraph": "In an interview published by Vogue in April 2013, Beyoncé was asked if she considers herself a feminist... She has also contributed to the Ban Bossy campaign..."
}
```

## Example Usage

```python
import pandas as pd

df = pd.read_csv("Squad2_ambious.csv")
print(df[['question', 'answer', 'info']].head())
```

## License and Release Policy

- Public datasets respect their original licenses.
- UniQA is institution-specific and released for research use after paper acceptance.
  
## References
[1] Pranav Rajpurkar, Robin Jia, and Percy Liang. 2018. Know What You Don’t Know: Unanswerable Questions for SQuAD. In *Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers)*, pages 784–789, Melbourne, Australia. Association for Computational Linguistics.

[2] Kiet Nguyen, Son Quoc Tran, Luan Thanh Nguyen, Tin Van Huynh, Son Thanh Luu, and Ngan Luu-Thuy Nguyen. 2022. VLSP 2021 – ViMRC Challenge: Vietnamese Machine Reading Comprehension. *VNU Journal of Science: Computer Science and Communication Engineering*, 38(2).

[3] Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William Cohen, Ruslan Salakhutdinov, and Christopher D. Manning. 2018. HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering. In *Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing*, pages 2369–2380, Brussels, Belgium. Association for Computational Linguistics.

[4] Khang Le, Hien Nguyen, Tung Le Thanh, and Minh Nguyen. 2022. VIMQA: A Vietnamese Dataset for Advanced Reasoning and Explainable Multi-hop Question Answering. In *Proceedings of the Thirteenth Language Resources and Evaluation Conference*, pages 6521–6529, Marseille, France. European Language Resources Association.
