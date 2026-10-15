#!/usr/bin/env python
"""Generate every Word document, the README results block and PROJECT_STATUS.md from results/.

    python scripts/generate_report_assets.py

All numbers are read from files written by the pipeline (see scripts/results_loader.py). Whatever has not
been produced yet is printed as "Pending execution" - nothing is invented. Re-run any time.
"""
from __future__ import annotations

import re
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import doc_content as DC  # noqa: E402
from scripts import viva_content as VC  # noqa: E402
from scripts.docx_helpers import Doc  # noqa: E402
from scripts.results_loader import Results  # noqa: E402
from src import config  # noqa: E402

PENDING = DC.PENDING
FIG = config.FIGURES_DIR
META = ["Machine Learning (3CS101CC24) - Nirma University, Institute of Technology - B.Tech CSE, Semester 5",
        "Student: [name / roll number - add before submission]"]


def banner(R: Results) -> str | None:
    if R.synthetic:
        return "SYNTHETIC SMOKE-TEST OUTPUT - NOT RESULTS OF THE PROJECT. Do not submit."
    if not R.has_results:
        return ("Status: the GTSRB experiment has not been executed yet - every measured value below reads 'Pending execution'. "
                "Run `python scripts/run_pipeline.py` and then `python scripts/generate_report_assets.py` to fill this document with real results.")
    mode = (R.audit or {}).get("run_mode", "")
    if mode != "full_dataset":
        return "DEVELOPMENT SUBSET - these numbers are NOT the full-dataset experiment."
    return None


def render(d: Doc, blocks, base_level: int = 1):
    for b in blocks:
        k = b[0]
        if k == "h":
            d.h(b[2], min(3, b[1] + base_level - 1))
        elif k == "p":
            d.p(b[1])
        elif k == "b":
            d.bullets(b[1])
        elif k == "t":
            d.table(b[1], b[2], caption=b[3] if len(b) > 3 else None)
        elif k == "img":
            d.figure(FIG / b[1], b[2])
        elif k == "note":
            d.callout(b[1])


def new_doc(R, title, subtitle="RoadEye - Traffic Sign Recognition Using HOG Features and SVM"):
    meta = META + [f"Run scope: {(R.audit or {}).get('run_mode', PENDING)}   |   Generated: {date.today().isoformat()}"]
    return Doc(title, subtitle, banner(R), meta)


def decisions(d: Doc, R, level=2):
    for title, what, why, how, changed, validated in DC.technical_decisions(R):
        d.h(title, level)
        d.table(["Question", "Answer"], [["WHAT?", what], ["WHY?", why], ["HOW?", how], ["WHAT CHANGED?", changed],
                                         ["HOW WAS IT VALIDATED?", validated]], [4, 12.5], font_size=9)


def split_blocks(blocks, markers):
    """Cut a block list at the h2 headings listed in markers (each marker starts a new part)."""
    parts, cur = [], []
    for b in blocks:
        if b[0] == "h" and any(b[2].startswith(m) for m in markers) and cur:
            parts.append(cur)
            cur = []
        cur.append(b)
    parts.append(cur)
    return parts


def only(blocks, *headings):
    out, on = [], False
    for b in blocks:
        if b[0] == "h":
            on = any(b[2].startswith(h) for h in headings)
        if on:
            out.append(b)
    return out


def abstract(R):
    base = ("RoadEye classifies cropped German traffic signs (GTSRB, 43 classes) with classical machine learning. A leakage-safe pipeline "
            "(sign-grouped splits, scaler fitted on training data only, official test set untouched until the end) compares raw pixels with HOG "
            "features (RQ1) and KNN, Decision Tree and SVM classifiers on identical HOG features (RQ2), tunes them with GridSearchCV and evaluates "
            "the final tuned HOG + SVM once on the test set. ")
    if not R.has_results:
        return base + f"Results: {PENDING} (this report regenerates automatically after the pipeline has been run)."
    t = R.final["test"]
    return base + (f"Final test accuracy {t['accuracy']:.4f}, macro-F1 {t['f1_macro']:.4f} on {R.final['n_test_samples']:,} test images. "
                   + DC.rq1_statement(R) + " " + DC.rq2_statement(R) + " The system is an educational study and is not safety-certified.")


STAGE_TABLE = [
    ["1", "Problem definition & use case", "Applicable", "Section 3"], ["2", "Data acquisition & understanding", "Applicable", "Section 4"],
    ["3", "Data preparation", "Applicable (corrupt/label/duplicate checks)", "Section 5"],
    ["4", "Data preprocessing", "Applicable (grayscale, resize, scaling; no missing values/categoricals - justified)", "Section 6"],
    ["5", "Data balancing", "Applicable - measured, decided by validation experiment", "Section 7"], ["6", "EDA", "Applicable", "Section 8"],
    ["7", "Feature engineering / selection", "Applicable (HOG; selection deliberately not used)", "Section 9"],
    ["8", "Baseline model", "Applicable", "Section 10"], ["9", "ML modelling", "Applicable (KNN, Decision Tree, SVM)", "Sections 11-12"],
    ["10", "Hyper-parameter tuning", "Applicable (GridSearchCV)", "Section 13"], ["11", "Validation & evaluation", "Applicable", "Sections 14-16"],
    ["12", "Use case, impact & limitations", "Applicable", "Sections 17-20"]]

REFERENCES = [
    "J. Stallkamp, M. Schlipsing, J. Salmen, C. Igel. Man vs. computer: Benchmarking machine learning algorithms for traffic sign recognition. Neural Networks 32 (2012) 323-332.",
    "N. Dalal, B. Triggs. Histograms of oriented gradients for human detection. CVPR 2005.",
    "C. Cortes, V. Vapnik. Support-vector networks. Machine Learning 20 (1995) 273-297.",
    "F. Pedregosa et al. Scikit-learn: Machine learning in Python. JMLR 12 (2011) 2825-2830.",
    "S. van der Walt et al. scikit-image: image processing in Python. PeerJ 2:e453 (2014).",
    "T. Mitchell. Machine Learning. TMH (course text book).",
    "Course policy and IA guidelines: Machine Learning 3CS101CC24, Nirma University (2025-26)."]


def integrity(d: Doc, R):
    d.p("**Dataset.** GTSRB, official archives (see DATASET_DOWNLOAD.md); the licence/usage terms of the copy actually downloaded must be confirmed on its page (LICENSE_NOTICE.md).")
    d.p("**Reference repositories** were read as material only. No code, model or result from them is part of this project or presented as ours. "
        "The code, experiments, tuning, evaluation and results are independently produced by running this repository.")
    rows = []
    for r in (R.ref or {}).get("repositories", []):
        rows.append([r["name"], r["commit"][:8], "no" if not r.get("model_file_present") else "present, not used",
                     r.get("license_file", ""), r["observation"]])
    d.table(["Repository", "Commit inspected", "Pretrained model used?", "Licence file", "Observation"],
            rows or [[PENDING] * 5], [3.6, 1.8, 2.2, 1.8, 7.2], font_size=8)
    d.p((R.ref or {}).get("statement", PENDING), italic=True)
    d.p("**Use of AI assistance.** AI-assisted tools were used for coding and documentation support. The student remains responsible for verifying and "
        "being able to explain every part of the project (IA guidelines, section 9). [Edit this statement to match your own use before submitting.]")


def build_stage_docs(R, out):
    S = DC.stage_blocks_all(R)
    made = []
    for n, (fname, title) in DC.DOC_FILES.items():
        d = new_doc(R, title)
        render(d, S[n], 1)
        made.append(d.save(out / f"{fname}.docx"))
    return made


def build_final_report(R, out):
    S = DC.stage_blocks_all(R)
    d = new_doc(R, "RoadEye - Final Report", "Traffic Sign Recognition Using HOG Features and SVM - Machine Learning Innovative Assignment")
    s9 = split_blocks(S[9], ["Fairness of the comparison"])
    s11 = split_blocks(S[11], ["Final metrics", "Error analysis"])
    ch = []  # (title, writer)
    d_ = lambda title, blocks: (title, lambda: render(d, blocks, 2))  # noqa: E731
    ch = [("Abstract", lambda: d.p(abstract(R))),
          ("Introduction", lambda: (d.p("This report follows the twelve-stage pipeline of the IA guidelines. Each stage is listed with its applicability; every major decision is "
                                        "justified in Appendix A (what / why / how / what changed / how validated)."),
                                    d.table(["#", "Stage", "Applicable?", "Where"], STAGE_TABLE, [0.8, 5, 7.5, 3], font_size=8.5),
                                    d.p("Central story: GTSRB -> preprocessing -> raw-pixel baseline -> HOG -> KNN / Decision Tree / SVM -> SVM GridSearchCV -> "
                                        "final evaluation -> error analysis -> Streamlit application."))),
          d_("Problem Definition", S[1]), d_("Dataset", S[2]), d_("Data Preparation", S[3]), d_("Preprocessing", S[4]),
          d_("Data Balancing", S[5]), d_("EDA", S[6]), d_("Feature Engineering", S[7]), d_("Baseline", S[8]),
          d_("Model Development", s9[0]), d_("Model Comparison", s9[1] if len(s9) > 1 else []), d_("Hyperparameter Tuning", S[10]),
          d_("Validation", s11[0]), d_("Final Evaluation", s11[1] if len(s11) > 1 else []), d_("Error Analysis", s11[2] if len(s11) > 2 else []),
          ("Application", lambda: (d.p("A Streamlit app (app/app.py) loads models/svm/best_svm_model.joblib, scaler.joblib and model_metadata.json, applies the same "
                                      "preprocessing + HOG functions as training, predicts the class and shows the top-3 SVM decision scores (explicitly not probabilities) with a safety "
                                      "disclaimer. Launch: `streamlit run app/app.py`."), render(d, only(S[12], "Practical use cases"), 2))),
          d_("Impact", only(S[12], "Benefits", "Stakeholders", "Risks", "Ethical")),
          d_("Limitations", only(S[12], "Limitations", "Why this system")), d_("Future Scope", only(S[12], "Future work")),
          ("Conclusion", lambda: (d.p(DC.rq1_statement(R)), d.p(DC.rq2_statement(R)),
                                  d.p("All values above were produced by running the code; the test set was evaluated once after the models were frozen."))),
          ("References", lambda: d.numbered(REFERENCES)),
          ("External Resources / Academic Integrity Declaration", lambda: integrity(d, R))]
    for i, (name, fn) in enumerate(ch, 1):
        d.h(f"{i}. {name}", 1)
        fn()
    d.h("Appendix A. Technical Decisions and Justifications", 1)
    decisions(d, R, 2)
    return d.save(out / "FINAL_REPORT.docx")


def build_presentation_and_viva(R, out):
    s = new_doc(R, "RoadEye - Presentation Outline", "Slide-by-slide outline (15 slides)")
    for i, (title, bullets, notes, fig) in enumerate(VC.slides(R), 1):
        s.h(f"Slide {i}: {title}", 2)
        s.bullets(bullets)
        s.p(f"Speaker note: {notes}", italic=True)
        if fig:
            s.figure(FIG / fig, f"Suggested visual: {fig}", 12)
    v = new_doc(R, "RoadEye - Viva Questions and Answers", "Likely examiner questions with concise answers")
    n = 0
    for cat, qa in VC.viva(R):
        v.h(cat, 2)
        for q, a in qa:
            n += 1
            v.p(f"Q{n}. {q}", bold=True)
            v.p(a)
    return [s.save(out / "ROAD_EYE_PRESENTATION_OUTLINE.docx"), v.save(out / "VIVA_QUESTIONS_AND_ANSWERS.docx")], n


def results_markdown(R) -> str:
    if not R.has_results or R.synthetic:
        return "[GENERATED AFTER RUNNING PIPELINE]\n\nRun `python scripts/run_pipeline.py`; this block is then replaced by the measured results."
    mode = R.audit["run_mode"]
    rows = ["| Model | Features | Accuracy | Macro-F1 | Training time (s) |", "|---|---|---|---|---|"]
    for r in R.cmp_val.itertuples():
        rows.append(f"| {r.model} | {r.feature_representation} | {r.accuracy:.4f} | {r.f1_macro:.4f} | {r.training_time_seconds:.1f} |")
    t = R.final["test"]
    out = [f"_Run mode: {mode}._", "", "**Validation set (model selection):**", ""] + rows + ["", "* " + DC.rq1_statement(R), "* " + DC.rq2_statement(R), "",
           "**Final test set (evaluated once, tuned HOG + SVM):**", "", "| Accuracy | Macro-precision | Macro-recall | Macro-F1 | Test images |", "|---|---|---|---|---|",
           f"| {t['accuracy']:.4f} | {t['precision_macro']:.4f} | {t['recall_macro']:.4f} | {t['f1_macro']:.4f} | {R.final['n_test_samples']:,} |"]
    return "\n".join(out)


def update_readme(R):
    p = ROOT / "README.md"
    if p.exists():
        new = f"<!-- RESULTS:START -->\n{results_markdown(R)}\n<!-- RESULTS:END -->"
        p.write_text(re.sub(r"<!-- RESULTS:START -->.*?<!-- RESULTS:END -->", lambda m: new, p.read_text(), flags=re.S))


def update_status(R, zip_done=False):
    m, ex = config.METRICS_DIR, (lambda *ps: all(Path(x).exists() for x in ps))  # noqa: E731
    real = R.has_results and not R.synthetic and (R.audit or {}).get("run_mode") == "full_dataset"
    mark = lambda ok, partial=False: "[x]" if ok else ("[~]" if partial else "[ ]")  # noqa: E731
    cmp_models = set(R.cmp_val["model"]) if R.cmp_val is not None else set()
    docs = ex(ROOT / "docs" / "FINAL_REPORT.docx")
    items = [
        ("Dataset acquired", mark(real and ex(m / "dataset_audit.json")), "GTSRB found and indexed (dataset_audit.json)"),
        ("Dataset validated", mark(real and ex(m / "dataset_audit.json")), "corrupt / label / duplicate checks"),
        ("Dataset statistics generated", mark(real and ex(m / "dataset_audit.json", FIG / "class_distribution.png")), "audit + class distribution"),
        ("EDA completed", mark(real and ex(m / "eda_summary.json", FIG / "eda_similar_classes.png")), "figures in results/figures"),
        ("Preprocessing completed", mark(real and ex(m / "data_preparation.json")), "data_preparation.json, split_info.json"),
        ("HOG implemented", mark(True), "src/hog_features.py (unit-tested)"),
        ("Baseline completed", mark(real and "Raw Pixel SVM" in cmp_models), "baseline_svm.json"),
        ("KNN completed", mark(real and "HOG KNN" in cmp_models), "model_comparison.csv"),
        ("Decision Tree completed", mark(real and "HOG Decision Tree" in cmp_models), "model_comparison.csv"),
        ("SVM completed", mark(real and "HOG SVM" in cmp_models), "model_comparison.csv"),
        ("GridSearch completed", mark(real and ex(m / "gridsearch_svm.json")), "gridsearch_svm.json"),
        ("Final evaluation completed", mark(real and ex(m / "final_test_metrics.json")), "final_test_metrics.json"),
        ("Error analysis completed", mark(real and ex(m / "error_analysis.json")), "error_analysis.json"),
        ("Reference model inspected", mark(ex(config.MODELS_DIR / "reference" / "reference_inspection.json")), "inspected, NOT numerically compared - see EXTERNAL_RESOURCES.md"),
        ("Streamlit app completed", mark(ex(ROOT / "app" / "app.py")), "code complete; smoke-tested on a synthetic stand-in model only (no real model yet)"),
        ("Tests completed", mark(True), "pytest tests/ (run on synthetic fixtures)"),
        ("Final report generated", mark(docs and real, docs), "FINAL_REPORT.docx ([~] = generated with 'Pending execution' placeholders)"),
        ("Presentation outline generated", mark(ex(ROOT / "docs" / "ROAD_EYE_PRESENTATION_OUTLINE.docx") and real, ex(ROOT / "docs" / "ROAD_EYE_PRESENTATION_OUTLINE.docx")), "same note as the report"),
        ("Viva questions generated", mark(ex(ROOT / "docs" / "VIVA_QUESTIONS_AND_ANSWERS.docx") and real, ex(ROOT / "docs" / "VIVA_QUESTIONS_AND_ANSWERS.docx")), "measured numbers appear after the pipeline run"),
        ("ZIP generated", mark(zip_done), "RoadEye_Traffic_Sign_ML_Assignment.zip"),
    ]
    lines = ["# PROJECT STATUS", "", "Legend: `[x]` genuinely completed and evidenced by a file - `[~]` code/document exists but contains 'Pending execution' placeholders - `[ ]` not done.",
             "", f"Last generated: {date.today().isoformat()} (by scripts/generate_report_assets.py; marks are computed, do not edit by hand)", ""]
    lines += [f"{mk} {n} - {note}" for n, mk, note in items]
    pending = [n for n, mk, _ in items if mk != "[x]"]
    lines += ["", "## Pending", ""] + ([f"- {n}" for n in pending] if pending else ["Nothing pending."])
    if not real:
        lines += ["", "**Root cause:** the GTSRB dataset could not be downloaded in the environment where this package was built (the official hosts are blocked "
                  "by its network policy). Everything dataset-dependent is therefore PENDING, not faked. Follow DATASET_DOWNLOAD.md, then run "
                  "`python scripts/run_pipeline.py` - the marks and all documents update themselves."]
    (ROOT / "PROJECT_STATUS.md").write_text("\n".join(lines) + "\n")


def main(argv=None) -> int:
    argv = argv or sys.argv[1:]
    R = Results()
    out = config.DOCS_DIR
    out.mkdir(parents=True, exist_ok=True)
    if "--status-only" not in argv:
        made = build_stage_docs(R, out) + [build_final_report(R, out)]
        more, nq = build_presentation_and_viva(R, out)
        made += more
        print(f"[reports] wrote {len(made)} documents to {out} ({nq} viva questions; results available: {R.has_results})")
        update_readme(R)
    update_status(R, zip_done="--zip-done" in argv)
    return 0


if __name__ == "__main__":
    sys.exit(main())
