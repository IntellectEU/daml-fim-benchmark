# Private Evaluation Terms — Daml FIM Benchmark

Last updated: November 13, 2026

These Private Evaluation Terms (“**Terms**”) govern evaluations of machine-learning models on the private subset of the Daml FIM benchmark (the “**Evaluation Service**”), operated by IntellectEU NV (“**IntellectEU**”, “**we**”, “**us**” or “**our**”). You accept them by ticking the acceptance box in the “**Private evaluation request**” issue form, or by confirming acceptance in writing when you request an evaluation through our support channel. If you accept on behalf of an organization, you represent that you are authorized to do so, and “**you**” includes that organization. The Evaluation Service is intended solely for business and professional use.

How evaluations are performed is described in the Private Evaluation process description published at <https://github.com/intellecteu/daml-fim-benchmark/blob/main/PRIVATE_EVALUATION.md>, in the version in effect when you make your request (the “**Process Description**”). If these Terms and the Process Description conflict, these Terms prevail.

## 1. The Evaluation Service

1.1 On request, IntellectEU runs the Daml FIM benchmark harness with the model you identify (the “**Submitted Model**”) against a private test set derived from Daml codebases that IntellectEU cannot redistribute (the “**Private Benchmark Data**”), and sends you a report of aggregate metrics (the “**Metrics Report**”).

1.2 The Evaluation Service is free of charge for models of up to approximately 7 billion parameters with fp16 or fp8 weights; larger models are at IntellectEU’s discretion and may require a separate agreement.

1.3 Evaluations are best-effort, with no turnaround, availability or support commitment. IntellectEU may decline, stop or cancel any request at any stage. A run that fails, or that cannot be scored as described in the Process Description, produces no Metrics Report; IntellectEU may repeat it or tell you that the evaluation will not be completed.

## 2. Your Submitted Model

2.1 You identify the Submitted Model by a link to a Hugging Face repository at a full commit hash. For a private repository, you grant read access to the Hugging Face account IntellectEU names; we recommend you revoke it once the evaluation is completed or ended.

2.2 You represent and warrant that you are entitled to make the Submitted Model available for evaluation under these Terms without breaching any licence, confidentiality obligation, third-party right or applicable law.

2.3 You retain all rights in the Submitted Model. You grant IntellectEU a non-exclusive, royalty-free, non-transferable licence to download, temporarily store and run it solely to perform the requested evaluation, including any repeat run under Section 1.3. The licence ends when IntellectEU sends you the Metrics Report or tells you that the evaluation will not be completed.

2.4 The Submitted Model must meet the requirements in the Process Description and must not be designed to extract, memorize or reveal the Private Benchmark Data or to interfere with the Evaluation Service.

2.5 You will indemnify IntellectEU against any third-party claim, and related losses and costs, arising from your breach of Section 2.2 or 2.4.

2.6 You represent that you are not, and are not owned or controlled by or acting on behalf of, a person subject to sanctions administered by the EU, the UN, the UK or the US.

## 3. Non-retention of the Submitted Model

3.1 IntellectEU will use the Submitted Model and the outputs it generates on the Private Benchmark Data (the “**Predictions**”) only to produce the Metrics Report, not to train, fine-tune or improve any model, and will not disclose them to any third party other than Google Cloud as its service provider under Section 3.4.

3.2 IntellectEU performs each evaluation as described in the Process Description. In particular, the Submitted Model and the Predictions are stored only on the local storage of a virtual machine created for that evaluation; only aggregate results, and operational progress and error messages, are taken off that machine; and the machine is deleted with its local storage when the evaluation ends, whether it completes, fails or is rejected. Each such machine is also set to be deleted automatically by Google Cloud two hours after it starts.

3.3 While an evaluation runs, the IntellectEU personnel who operate the Evaluation Service have administrative access to the virtual machine. The commitments in this Section therefore rest on IntellectEU’s process and its personnel’s confidentiality obligations, not solely on technical measures.

3.4 The virtual machine runs on Google Cloud as IntellectEU’s service provider, and the physical erasure of deleted storage is performed by Google Cloud under its own terms. Copies held by Hugging Face or any other third party are outside IntellectEU’s control.

## 4. Private Benchmark Data

You receive no access to the Private Benchmark Data, Predictions or per-sample results, and you must not attempt to reconstruct or infer the Private Benchmark Data from Metrics Reports or by any other means.

## 5. What IntellectEU keeps

5.1 For each evaluation, IntellectEU keeps the Metrics Report (if any), the aggregate summary it is produced from, and a record of the evaluation: the Submitted Model’s repository and commit, the evaluation settings and software versions, the request, the operator and the date. These records contain no part of the Submitted Model’s weights or the Predictions. IntellectEU keeps them for 90 days. Copies of the Metrics Report sent to you, and our correspondence with you, are kept as set out in Section 6.

5.2 You may use and publish your Metrics Report. IntellectEU will not publish your Metrics Report, or the fact that you requested an evaluation, without your consent, except to the extent you have made it public yourself (for example through a public request issue).

## 6. Personal Data

IntellectEU processes the Personal Data involved in a request as described in the Privacy Notice published at <https://api.dca.catalyx.solutions/privacy>. That notice is provided for information and does not form part of these Terms.

## 7. Disclaimer and limitation of liability

7.1 TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THE EVALUATION SERVICE AND METRICS REPORTS ARE PROVIDED “AS IS”, WITHOUT ANY WARRANTY, INCLUDING AS TO ACCURACY, COMPLETENESS OR FITNESS FOR A PARTICULAR PURPOSE. A Metrics Report measures the Submitted Model on this benchmark only.

7.2 TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, INTELLECTEU WILL NOT BE LIABLE FOR INDIRECT OR CONSEQUENTIAL DAMAGES OR LOSS OF PROFITS, REVENUE OR BUSINESS, AND ITS TOTAL AGGREGATE LIABILITY ARISING OUT OF OR RELATING TO THE EVALUATION SERVICE OR THESE TERMS WILL NOT EXCEED EUR 100. These limitations do not apply to liability that cannot lawfully be excluded or limited, including for fraud, wilful misconduct, or death or personal injury caused by negligence.

## 8. General

8.1 IntellectEU may modify, suspend or discontinue the Evaluation Service, and may update these Terms and the Process Description; updates apply to requests made after they are published.

8.2 A separate written agreement with IntellectEU covering an evaluation prevails over these Terms to the extent of any conflict. An invalid provision is limited to the minimum extent necessary and the rest remains effective.

8.3 These Terms are governed by Belgian law, and the courts of Brussels, Belgium have exclusive jurisdiction, to the maximum extent permitted by applicable law.

8.4 Sections 2.2, 2.4, 2.5, 4, 5.2 and 7 survive the end of any evaluation.

## 9. Contact

- Questions about the Evaluation Service: <https://docs.catalyx.solutions/support-center/contact-support>
- Questions concerning these Terms: <legal@intellecteu.com>
- Privacy questions and data-subject requests: <privacy@intellecteu.com>
