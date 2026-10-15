#!/usr/bin/env python3
"""
Generate a comprehensive, publication-grade 20+ page PDF Report and Viva Defense Guide
for the RoadEye Machine Learning Innovative Assignment.

Complies strictly with Guidelines_Machine_Learning_innovative_Assignment.pdf:
- Covers all 12 stages in rigorous detail (What, Why, How, Impact, Validation)
- Includes full theoretical formulations and mathematical derivations
- Embeds high-resolution visualization figures from results/figures/
- Features exact measured experimental tables and metrics
- Provides a step-by-step live demonstration walkthrough for the professor
- Contains a 35-question Viva Defense mastercompendium
"""

import os
import sys
import base64
import subprocess
import re

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DOCS_DIR = os.path.join(BASE_DIR, "docs")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
FIG_DIR = os.path.join(RESULTS_DIR, "figures")
HTML_OUT = os.path.join(DOCS_DIR, "ROAD_EYE_MASTER_REPORT.html")
PDF_OUT = os.path.join(DOCS_DIR, "ROAD_EYE_COMPLETE_MASTER_GUIDE_20PAGES.pdf")

def get_base64_img(filename):
    path = os.path.join(FIG_DIR, filename)
    if os.path.exists(path):
        with open(path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")
            return f"data:image/png;base64,{encoded}"
    return ""

def build_html():
    img_map = {
        "__IMG_PIPELINE_FLOW__": get_base64_img("raw_vs_hog_features.png"),
        "__IMG_HOG_VIS__": get_base64_img("hog_visualization.png"),
        "__IMG_CLASS_DIST__": get_base64_img("class_distribution.png"),
        "__IMG_CLASS_DIST_TRAIN__": get_base64_img("class_distribution_train_split.png"),
        "__IMG_SAMPLE_IMAGES__": get_base64_img("sample_images.png"),
        "__IMG_SIZE_DIST__": get_base64_img("image_size_distribution.png"),
        "__IMG_CLASS_MEANS__": get_base64_img("eda_class_mean_images.png"),
        "__IMG_SIMILAR_CLASSES__": get_base64_img("eda_similar_classes.png"),
        "__IMG_BRIGHTNESS__": get_base64_img("eda_brightness_contrast.png"),
        "__IMG_SIZE_BY_CLASS__": get_base64_img("eda_size_by_class.png"),
        "__IMG_BASELINE_CM__": get_base64_img("baseline_confusion_matrix.png"),
        "__IMG_KNN_CM__": get_base64_img("knn_default_confusion_matrix.png"),
        "__IMG_KNN_TUNED_CM__": get_base64_img("knn_tuned_confusion_matrix.png"),
        "__IMG_TREE_CM__": get_base64_img("decision_tree_default_confusion_matrix.png"),
        "__IMG_TREE_TUNED_CM__": get_base64_img("decision_tree_tuned_confusion_matrix.png"),
        "__IMG_SVM_CM__": get_base64_img("svm_default_confusion_matrix.png"),
        "__IMG_SVM_TUNED_CM__": get_base64_img("svm_tuned_confusion_matrix.png"),
        "__IMG_PERF_COMP__": get_base64_img("model_performance_comparison.png"),
        "__IMG_PERF_COMP_TEST__": get_base64_img("model_performance_comparison_test.png"),
        "__IMG_GRID_SEARCH__": get_base64_img("grid_search_svm.png"),
        "__IMG_FINAL_CM__": get_base64_img("final_confusion_matrix.png"),
        "__IMG_FINAL_F1__": get_base64_img("final_per_class_f1.png"),
        "__IMG_MISCLASSIFIED__": get_base64_img("misclassified_examples.png"),
    }

    template = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>RoadEye: Complete Technical Report, Theory &amp; Viva Guide</title>
<style>
  @page {
    size: A4;
    margin: 18mm 16mm 20mm 16mm;
    @bottom-right {
      content: "Page " counter(page);
      font-size: 8pt;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      color: #64748b;
    }
    @bottom-left {
      content: "RoadEye: Traffic Sign Recognition (GTSRB) | Course 3CS101CC24";
      font-size: 8pt;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      color: #64748b;
    }
  }

  body {
    font-family: "Segoe UI", -apple-system, BlinkMacSystemFont, Roboto, "Helvetica Neue", Arial, sans-serif;
    color: #1e293b;
    line-height: 1.52;
    font-size: 9.6pt;
    margin: 0;
    padding: 0;
  }

  .page-break {
    page-break-before: always;
  }

  /* Typography */
  h1, h2, h3, h4 {
    color: #0f172a;
    font-weight: 700;
    margin-top: 1.1em;
    margin-bottom: 0.35em;
    page-break-after: avoid;
  }

  h1 {
    font-size: 18pt;
    border-bottom: 2.5px solid #2563eb;
    padding-bottom: 4px;
    margin-top: 0;
    letter-spacing: -0.02em;
  }

  h2 {
    font-size: 13pt;
    border-left: 4px solid #2563eb;
    padding-left: 8px;
    margin-top: 1.3em;
    background: #f8fafc;
    padding-top: 3px;
    padding-bottom: 3px;
  }

  h3 {
    font-size: 10.8pt;
    color: #1e40af;
    margin-top: 0.9em;
  }

  h4 {
    font-size: 9.8pt;
    color: #334155;
    margin-top: 0.7em;
  }

  p {
    margin: 0.45em 0;
    text-align: justify;
  }

  /* Cover Page */
  .cover-page {
    text-align: center;
    padding-top: 40px;
    page-break-after: always;
  }

  .cover-badge {
    display: inline-block;
    background: #eff6ff;
    color: #1d4ed8;
    border: 1px solid #bfdbfe;
    padding: 5px 14px;
    border-radius: 9999px;
    font-weight: 600;
    font-size: 9.5pt;
    margin-bottom: 16px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }

  .cover-title {
    font-size: 25pt;
    font-weight: 800;
    color: #0f172a;
    line-height: 1.2;
    margin: 12px 0 8px 0;
    letter-spacing: -0.03em;
  }

  .cover-subtitle {
    font-size: 12pt;
    color: #475569;
    font-weight: 500;
    max-width: 620px;
    margin: 0 auto 25px auto;
    line-height: 1.45;
  }

  .cover-divider {
    width: 80px;
    height: 4px;
    background: #2563eb;
    margin: 25px auto;
    border-radius: 2px;
  }

  .meta-box {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 16px 22px;
    max-width: 520px;
    margin: 20px auto;
    text-align: left;
    font-size: 9.2pt;
  }

  .meta-row {
    display: flex;
    justify-content: space-between;
    padding: 4px 0;
    border-bottom: 1px solid #f1f5f9;
  }

  .meta-row:last-child {
    border-bottom: none;
  }

  .meta-label {
    color: #64748b;
    font-weight: 600;
  }

  .meta-val {
    color: #0f172a;
    font-weight: 700;
  }

  .cover-abstract {
    margin: 30px auto 0 auto;
    max-width: 620px;
    background: #ffffff;
    border-left: 3px solid #2563eb;
    padding: 12px 16px;
    font-size: 8.8pt;
    text-align: justify;
    color: #334155;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
  }

  /* Callout Boxes */
  .callout {
    background: #f0fdf4;
    border-left: 4px solid #16a34a;
    padding: 8px 12px;
    margin: 10px 0;
    font-size: 8.9pt;
    border-radius: 0 6px 6px 0;
  }

  .callout-title {
    font-weight: 700;
    color: #15803d;
    margin-bottom: 3px;
  }

  .callout-warn {
    background: #fefce8;
    border-left: 4px solid #ca8a04;
  }

  .callout-warn .callout-title {
    color: #a16207;
  }

  .callout-info {
    background: #eff6ff;
    border-left: 4px solid #2563eb;
  }

  .callout-info .callout-title {
    color: #1d4ed8;
  }

  /* Tables */
  table {
    width: 100%;
    border-collapse: collapse;
    margin: 10px 0;
    font-size: 8.3pt;
    page-break-inside: avoid;
  }

  th {
    background: #1e293b;
    color: #ffffff;
    padding: 6px 8px;
    text-align: left;
    font-weight: 600;
    font-size: 8.3pt;
  }

  td {
    padding: 5px 8px;
    border-bottom: 1px solid #e2e8f0;
    color: #1e293b;
  }

  tr:nth-child(even) td {
    background: #f8fafc;
  }

  .best-row td {
    background: #ecfdf5 !important;
    font-weight: 700;
    color: #065f46;
  }

  /* Math and Formulas */
  .math-block {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    padding: 8px 12px;
    margin: 8px 0;
    border-radius: 6px;
    font-family: "Cambria Math", "Times New Roman", serif;
    font-size: 9.8pt;
    text-align: center;
    color: #0f172a;
  }

  .formula-expl {
    font-size: 8.6pt;
    color: #475569;
    margin-top: 3px;
    text-align: left;
  }

  /* Figures */
  .figure-box {
    margin: 10px 0;
    text-align: center;
    page-break-inside: avoid;
  }

  .figure-img {
    max-width: 95%;
    max-height: 220px;
    border: 1px solid #e2e8f0;
    border-radius: 6px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
  }

  .figure-caption {
    font-size: 8.1pt;
    color: #64748b;
    margin-top: 4px;
    font-weight: 600;
  }

  /* 2-column layout */
  .grid-2 {
    display: flex;
    gap: 12px;
    margin: 8px 0;
  }

  .grid-2 > div {
    flex: 1;
  }

  /* Viva Cards */
  .viva-item {
    background: #ffffff;
    border: 1px solid #cbd5e1;
    border-radius: 6px;
    padding: 8px 11px;
    margin-bottom: 8px;
    page-break-inside: avoid;
  }

  .viva-q {
    font-weight: 700;
    color: #1e3a8a;
    font-size: 9.1pt;
    margin-bottom: 3px;
  }

  .viva-q::before {
    content: "Q: ";
    color: #2563eb;
    font-weight: 800;
  }

  .viva-a {
    color: #334155;
    font-size: 8.6pt;
    line-height: 1.45;
  }

  .viva-a::before {
    content: "Answer: ";
    font-weight: 700;
    color: #047857;
  }

  /* Code Block */
  .code-snip {
    background: #0f172a;
    color: #f8fafc;
    font-family: "Courier New", Courier, monospace;
    font-size: 8pt;
    padding: 7px 10px;
    border-radius: 5px;
    margin: 6px 0;
    overflow-x: hidden;
    line-height: 1.35;
  }

</style>
</head>
<body>

<!-- ========================================================================= -->
<!-- COVER PAGE                                                                -->
<!-- ========================================================================= -->
<div class="cover-page">
  <div class="cover-badge">Course 3CS101CC24 &bull; Innovative Assignment Comprehensive Compendium</div>
  <div class="cover-title">RoadEye: Classical ML Traffic Sign Recognition Pipeline</div>
  <div class="cover-subtitle">
    An End-to-End, Leakage-Safe, Rigorously Evaluated Machine Learning Solution using HOG Feature Engineering and Support Vector Machines on the German Traffic Sign Recognition Benchmark (GTSRB)
  </div>

  <div class="cover-divider"></div>

  <div class="meta-box">
    <div class="meta-row">
      <span class="meta-label">Application Domain</span>
      <span class="meta-val">Transportation, Computer Vision &amp; Smart Systems</span>
    </div>
    <div class="meta-row">
      <span class="meta-label">Dataset Benchmark</span>
      <span class="meta-val">GTSRB (51,839 Total Images across 43 Classes)</span>
    </div>
    <div class="meta-row">
      <span class="meta-label">Proposed Architecture</span>
      <span class="meta-val">Histogram of Oriented Gradients (HOG) + Tuned RBF SVM</span>
    </div>
    <div class="meta-row">
      <span class="meta-label">Baseline Reference</span>
      <span class="meta-val">Raw Grayscale Pixels (1,024-d) + RBF SVM</span>
    </div>
    <div class="meta-row">
      <span class="meta-label">Comparative Models</span>
      <span class="meta-val">Raw SVM, HOG KNN, HOG Decision Tree, HOG SVM</span>
    </div>
    <div class="meta-row">
      <span class="meta-label">Final Test Score</span>
      <span class="meta-val">94.08% Accuracy &bull; 0.9256 Macro-F1 (12,630 Test Samples)</span>
    </div>
    <div class="meta-row">
      <span class="meta-label">Deployment Interface</span>
      <span class="meta-val">Interactive Streamlit Web Dashboard (Inference &amp; Safety Margins)</span>
    </div>
  </div>

  <div class="cover-abstract">
    <strong>Abstract &amp; Scope:</strong> This document serves as the complete technical report, theoretical justification, demonstration manual, and viva defense guide for the RoadEye traffic-sign recognition system. Built to satisfy all 12 mandatory stages of the Nirma University Machine Learning Innovative Assignment guidelines, this project addresses two formal research questions: <em>RQ1: Does HOG edge-orientation feature engineering outperform raw pixel intensity inputs?</em> and <em>RQ2: Which classical supervised classifier (KNN, Decision Tree, or SVM) provides optimal generalization and computational balance?</em> The pipeline enforces rigorous data-hygiene, prevents near-duplicate leakage via physical sign-track grouping, utilizes stratified cross-validation for hyperparameter tuning, evaluates the official GTSRB test split exactly once, and provides a production-ready Streamlit web interface.
  </div>
</div>

<!-- ========================================================================= -->
<!-- TABLE OF CONTENTS & QUICK EXECUTIVE BRIEFING                              -->
<!-- ========================================================================= -->
<div class="page-break"></div>
<h1>Executive Briefing &amp; Quick Reference</h1>

<div class="callout callout-info">
  <div class="callout-title">The 2-Minute Elevator Pitch for Your Professor</div>
  <p>
    <em>"Professor, RoadEye is an end-to-end classical machine learning system that classifies German traffic signs into 43 categories. We strictly adhered to the 12-stage curriculum pipeline. Rather than treating ML as a black-box library call, we designed a reproducible, leakage-safe pipeline addressing two research questions: First, does Histogram of Oriented Gradients (HOG) feature engineering beat raw pixels? Second, which classical algorithm performs best on these features among KNN, Decision Trees, and SVM? By resizing images to 32x32, computing 1,764 HOG descriptors, preventing sign-track data leakage, and tuning via Stratified Group 3-Fold Cross-Validation, our Tuned HOG-SVM achieved <strong>94.08% accuracy</strong> and a <strong>0.9256 Macro-F1 score</strong> on the untouched 12,630 test images&mdash;outperforming the raw pixel baseline by <strong>+25.82% Macro-F1</strong>. Finally, we deployed the model into an interactive Streamlit application displaying real-time predictions and decision margins."</em>
  </p>
</div>

<h2>Master Performance Comparison (All Models on Untouched Test Split)</h2>
<p>
  Below is the empirical summary of all evaluated configurations on the official GTSRB test set (N = 12,630 unseen images). All numbers were computed deterministically by the automated pipeline (seed 42).
</p>

<table>
  <thead>
    <tr>
      <th>Model Architecture</th>
      <th>Feature Space</th>
      <th>Accuracy</th>
      <th>Macro-Precision</th>
      <th>Macro-Recall</th>
      <th>Macro-F1</th>
      <th>Balanced Acc.</th>
      <th>Inference / Sample</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>Raw Pixel SVM (Baseline)</strong></td>
      <td>Raw Pixels (1,024-d)</td>
      <td>72.27%</td>
      <td>0.6967</td>
      <td>0.6763</td>
      <td>0.6664</td>
      <td>0.6763</td>
      <td>11.66 ms</td>
    </tr>
    <tr>
      <td><strong>HOG Decision Tree (Default)</strong></td>
      <td>HOG (1,764-d)</td>
      <td>74.96%</td>
      <td>0.7115</td>
      <td>0.6928</td>
      <td>0.6981</td>
      <td>0.6928</td>
      <td>&lt; 0.01 ms</td>
    </tr>
    <tr>
      <td><strong>Tuned HOG Decision Tree</strong></td>
      <td>HOG (1,764-d)</td>
      <td>73.70%</td>
      <td>0.7079</td>
      <td>0.6862</td>
      <td>0.6927</td>
      <td>0.6862</td>
      <td>&lt; 0.01 ms</td>
    </tr>
    <tr>
      <td><strong>HOG KNN (Default k=5)</strong></td>
      <td>HOG (1,764-d)</td>
      <td>88.37%</td>
      <td>0.8657</td>
      <td>0.8373</td>
      <td>0.8451</td>
      <td>0.8373</td>
      <td>0.39 ms</td>
    </tr>
    <tr>
      <td><strong>Tuned HOG KNN (k=7, distance)</strong></td>
      <td>HOG (1,764-d)</td>
      <td>88.50%</td>
      <td>0.8723</td>
      <td>0.8398</td>
      <td>0.8483</td>
      <td>0.8398</td>
      <td>0.39 ms</td>
    </tr>
    <tr>
      <td><strong>HOG SVM (Default C=1, RBF)</strong></td>
      <td>HOG (1,764-d)</td>
      <td>93.91%</td>
      <td>0.9508</td>
      <td>0.9063</td>
      <td>0.9246</td>
      <td>0.9063</td>
      <td>14.02 ms</td>
    </tr>
    <tr class="best-row">
      <td><strong>Tuned HOG SVM (PROPOSED)</strong></td>
      <td>HOG (1,764-d)</td>
      <td>94.08%</td>
      <td>0.9521</td>
      <td>0.9069</td>
      <td>0.9256</td>
      <td>0.9069</td>
      <td>14.02 ms</td>
    </tr>
  </tbody>
</table>

<div class="grid-2">
  <div class="callout callout-info">
    <div class="callout-title">Answer to Research Question 1 (RQ1)</div>
    <p>
      <strong>Does HOG beat raw pixels? YES!</strong> Under identical split and SVM hyperparameters, HOG features achieved <strong>0.9246 Macro-F1</strong> versus <strong>0.6664 Macro-F1</strong> for raw pixels. HOG provides a massive <strong>+0.2582 (+25.82%) Macro-F1 gain</strong> and <strong>+21.64% accuracy improvement</strong> because local gradient histograms provide illumination invariance and structural edge abstraction.
    </p>
  </div>
  <div class="callout callout-info">
    <div class="callout-title">Answer to Research Question 2 (RQ2)</div>
    <p>
      <strong>Which classifier wins on HOG? SVM!</strong> Across all evaluations, SVM decisively outperformed KNN and Decision Trees: <strong>Tuned SVM (0.9256) &gt; Tuned KNN (0.8483) &gt; Tuned Decision Tree (0.6927)</strong>. SVM excels because its maximum-margin hyperplane in RKHS handles the 1,764-dimensional feature space without suffering from the curse of dimensionality or greedy partitioning artifacts.
    </p>
  </div>
</div>

<div class="figure-box">
  <img src="__IMG_PERF_COMP_TEST__" class="figure-img" style="max-height: 190px;" alt="Model Performance Comparison">
  <div class="figure-caption">Figure 1: Empirical evaluation of Accuracy, Precision, Recall, and Macro-F1 across all models on the 12,630 test set.</div>
</div>

<!-- ========================================================================= -->
<!-- SECTION 1: ARCHITECTURE & PIPELINE OVERVIEW                              -->
<!-- ========================================================================= -->
<div class="page-break"></div>
<h1>1. Project Architecture &amp; Technical Foundations</h1>

<p>
  The architecture of RoadEye follows a modular, reproducible design implementing the full data-to-deployment lifecycle mandated by the syllabus. The system is split into distinct functional layers with complete separation of concerns:
</p>

<div class="figure-box">
  <img src="__IMG_PIPELINE_FLOW__" class="figure-img" style="max-height: 210px;" alt="Pipeline Flowchart">
  <div class="figure-caption">Figure 2: End-to-end data pipeline from raw PPM sign imagery through ROI cropping, HOG transformation, track-grouped splitting, cross-validation, and deployment.</div>
</div>

<h2>System Tech Stack &amp; Library Justifications</h2>
<ul>
  <li><strong>Core Language:</strong> Python 3.9+ &bull; Deterministic execution via fixed seeds (<code>SEED = 42</code>) across NumPy, Python random, and scikit-learn.</li>
  <li><strong>scikit-image (<code>skimage.feature.hog</code>):</strong> Industrial-standard HOG implementation featuring 9-bin orientation discretization, L2-Hysteresis block normalization, and square-root variance stabilization.</li>
  <li><strong>scikit-learn:</strong> Pipeline orchestration, <code>StandardScaler</code>, <code>StratifiedGroupKFold</code>, <code>GridSearchCV</code>, <code>SVC</code> (LIBSVM wrapper), <code>KNeighborsClassifier</code>, <code>DecisionTreeClassifier</code>, and comprehensive evaluation metrics.</li>
  <li><strong>OpenCV (<code>cv2</code>):</strong> Accelerated image I/O, ROI bounding-box cropping, and bilinear spatial resizing (32x32).</li>
  <li><strong>Pandas &amp; NumPy:</strong> High-performance vector operations, metadata parsing, class auditing, and CSV/JSON persistence.</li>
  <li><strong>Streamlit:</strong> Lightweight, reactive web application framework enabling live model inference, file upload, image inspection, and safety margin diagnostics.</li>
  <li><strong>pytest:</strong> Unit and regression testing suite utilizing synthetic image fixtures to validate pipeline integrity in under 10 seconds.</li>
</ul>

<h2>Why Classical Machine Learning Instead of Deep Learning (CNN)?</h2>
<p>
  A frequent question from evaluators is: <em>"Why did you use classical ML (HOG + SVM) when modern traffic sign recognition is dominated by Convolutional Neural Networks (CNNs)?"</em> The technical and pedagogical justifications are:
</p>
<ol>
  <li><strong>Syllabus Alignment &amp; Core Fundamentals:</strong> The course (3CS101CC24) emphasizes foundational ML paradigms: feature space engineering, mathematical optimization (convex quadratic programming), kernel tricks, margin theory, and non-parametric estimators.</li>
  <li><strong>Transparent, Interpretable Feature Space:</strong> Unlike deep neural networks with millions of uninterpretable weights, HOG creates a deterministic 1,764-dimensional descriptor grounded in physical edge orientations. Evaluators can directly inspect what features caused a classification.</li>
  <li><strong>Low Computational Footprint:</strong> Training our complete HOG-SVM pipeline takes approximately 2.5 minutes on a standard CPU, enabling full cross-validation and hyperparameter grid searches without requiring GPU clusters or cloud credits.</li>
  <li><strong>Formal Scientific Benchmark:</strong> GTSRB was originally introduced by Stallkamp et al. specifically to benchmark classical ML pipelines against human performance. Evaluating HOG against raw pixels provides rigorous hypothesis testing (RQ1).</li>
</ol>

<!-- ========================================================================= -->
<!-- STAGE 1: PROBLEM DEFINITION                                               -->
<!-- ========================================================================= -->
<div class="page-break"></div>
<h1>Stage 1: Problem Definition &amp; Use Case Specification</h1>

<h2>1.1 Real-World Problem Context</h2>
<p>
  Traffic Sign Recognition (TSR) is a critical perception component in Advanced Driver Assistance Systems (ADAS), autonomous vehicles, and automated municipal highway inventory systems. A vehicle navigating dynamic roadway environments must accurately detect, classify, and respond to regulatory signs (speed limits, stop, yield), warning signs (curves, pedestrian crossings, construction), and informational markers. Misclassifications or latency can lead to hazardous moving violations or fatal collisions.
</p>

<h2>1.2 System Inputs, Outputs &amp; Constraints</h2>
<ul>
  <li><strong>Input:</strong> Single-image cropped photographs of individual traffic signs captured by vehicle-mounted forward-facing cameras in varied real-world conditions (scale, blur, weather, shadows, occlusion).</li>
  <li><strong>Output:</strong> A discrete categorical prediction in {0, 1, ..., 42} representing one of the 43 standard German traffic sign classes, accompanied by ranked decision function margin scores.</li>
  <li><strong>Constraints:</strong> 
    (1) Classical ML syllabus constraints (no convolutional layers or deep backpropagation);
    (2) Strict zero-leakage protocol between consecutive video frames of identical physical signs;
    (3) Real-time CPU inference feasibility (&lt; 20 ms per image).
  </li>
  <li><strong>Expected Impact:</strong> Enables deterministic, lightweight driver notification systems on resource-constrained embedded edge processors (e.g., Raspberry Pi, automotive ECUs) without requiring high-power GPU coprocessors.</li>
</ul>

<h2>1.3 Formal Research Questions (Hypothesis Testing)</h2>
<div class="grid-2">
  <div class="callout callout-info">
    <div class="callout-title">Research Question 1 (RQ1)</div>
    <p>
      <em>"Does explicit feature engineering via Histogram of Oriented Gradients (HOG) significantly improve traffic sign classification accuracy and Macro-F1 over raw grayscale pixel intensity representations when using identical classifier architectures and split boundaries?"</em>
    </p>
  </div>
  <div class="callout callout-info">
    <div class="callout-title">Research Question 2 (RQ2)</div>
    <p>
      <em>"Among three diverse classical supervised learning paradigms&mdash;instance-based (KNN), tree-based hierarchical partitioning (Decision Trees), and maximum-margin hyperplane optimization (SVM)&mdash;which algorithm achieves optimal generalization on HOG descriptors?"</em>
    </p>
  </div>
</div>

<!-- ========================================================================= -->
<!-- STAGE 2: DATA ACQUISITION & UNDERSTANDING                                 -->
<!-- ========================================================================= -->
<h2>Stage 2: Data Acquisition &amp; Understanding</h2>

<p>
  We utilize the official <strong>German Traffic Sign Recognition Benchmark (GTSRB)</strong> introduced by Stallkamp et al. (INNS/IEEE, Neural Networks 2012). The dataset represents a gold-standard multiclass benchmark captured across diverse German roadways under realistic operational conditions.
</p>

<table>
  <thead>
    <tr>
      <th>Dataset Property</th>
      <th>Specification / Measured Value</th>
      <th>Technical Implication</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>Total Files Audited</strong></td>
      <td>51,839 images (.ppm format)</td>
      <td>Comprehensive multiclass real-world sample</td>
    </tr>
    <tr>
      <td><strong>Training Pool (Train + Val)</strong></td>
      <td>39,209 images across 1,307 physical tracks</td>
      <td>Split ~80/20 grouped by physical track</td>
    </tr>
    <tr>
      <td><strong>Official Test Set</strong></td>
      <td>12,630 images (untouched final evaluation)</td>
      <td>Evaluated exactly once at project conclusion</td>
    </tr>
    <tr>
      <td><strong>Number of Target Classes</strong></td>
      <td>43 distinct categories (Classes 0 to 42)</td>
      <td>Multiclass classification problem</td>
    </tr>
    <tr>
      <td><strong>Image Resolutions</strong></td>
      <td>Variable: 25x25 min, 43x43 median, 266x266 max</td>
      <td>Mandates standardized spatial resizing (32x32)</td>
    </tr>
    <tr>
      <td><strong>Physical Track Structure</strong></td>
      <td>~30 consecutive video frames per physical sign</td>
      <td><strong>CRITICAL:</strong> High temporal similarity mandates group splitting</td>
    </tr>
    <tr>
      <td><strong>Class Imbalance Ratio</strong></td>
      <td>10.71 : 1 (Class 2: 2,250 vs Class 0/19/37: 210)</td>
      <td>Mandates Macro-F1 metric over raw accuracy</td>
    </tr>
  </tbody>
</table>

<div class="figure-box">
  <img src="__IMG_CLASS_DIST__" class="figure-img" style="max-height: 190px;" alt="Class Distribution">
  <div class="figure-caption">Figure 3: Class frequency distribution across all 43 classes in the GTSRB training pool, showing severe class imbalance (10.71x).</div>
</div>

<!-- ========================================================================= -->
<!-- STAGE 3 & 4: DATA PREPARATION & PREPROCESSING                            -->
<!-- ========================================================================= -->
<div class="page-break"></div>
<h1>Stage 3 &amp; 4: Data Preparation, Preprocessing &amp; Leakage Control</h1>

<h2>3.1 Data Preparation &amp; Integrity Auditing</h2>
<p>
  Before conducting modeling, an automated integrity audit script (src/data_validation.py) inspected all 51,839 image files and annotation tables. The audit verified:
</p>
<ul>
  <li><strong>Unreadable / Corrupted Files:</strong> Exactly 0 unreadable files detected; all PPM headers and byte buffers decoded correctly.</li>
  <li><strong>Label Inconsistencies:</strong> 0 mismatches between directory folder IDs and internal CSV annotation tables. Class IDs are strictly validated in [0, 42].</li>
  <li><strong>ROI Coordinate Integrity:</strong> Verified that for all records, 0 &le; X1 &lt; X2 &le; Width and 0 &le; Y1 &lt; Y2 &le; Height.</li>
  <li><strong>Exact Duplicate Detection:</strong> Byte-level MD5 hashing identified 8 exact duplicates between train and test pools (identical camera sensor double-reads); near-duplicates were properly isolated via track grouping.</li>
</ul>

<h2>4.1 Data Preprocessing Pipeline</h2>
<p>
  Raw photographs cannot be fed directly into classical algorithms due to arbitrary dimensions and background clutter. Preprocessing standardizes every sample through four sequential operations:
</p>

<div class="grid-2">
  <div>
    <h4>Step 1: ROI Bounding-Box Cropping</h4>
    <p>
      Vehicle-mounted camera captures include surrounding scenery (sky, foliage, asphalt). The official GTSRB ground truth provides bounding box coordinates (X1, Y1, X2, Y2). We crop the image to isolate the sign face, removing extraneous noise:
    </p>
    <div class="code-snip">cropped = image[y1:y2, x1:x2]</div>
  </div>
  <div>
    <h4>Step 2: Grayscale Luminance Conversion</h4>
    <p>
      Color can introduce severe variance under differing daylight, sunrise, or overcast conditions. HOG captures spatial gradient orientations which are structurally identical across color channels. We convert RGB to single-channel luminance:
    </p>
    <div class="code-snip">gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)</div>
  </div>
</div>

<div class="grid-2">
  <div>
    <h4>Step 3: Bilinear Spatial Normalization (32x32)</h4>
    <p>
      Images vary from 25x25 up to 266x266 pixels. Bilinear interpolation standardizes all inputs to a uniform spatial grid of 32x32 pixels, balancing edge resolution against feature vector dimensionality.
    </p>
    <div class="code-snip">resized = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_LINEAR)</div>
  </div>
  <div>
    <h4>Step 4: Standard Feature Scaling (StandardScaler)</h4>
    <p>
      HOG features and pixel intensities must have zero mean and unit variance for distance-based estimators (SVM, KNN). Crucially, <strong>the scaler is fit strictly on training splits</strong> and applied transform-only to validation/test sets to eliminate data leakage.
    </p>
    <div class="code-snip">X_scaled = scaler.transform(X_raw)</div>
  </div>
</div>

<div class="figure-box">
  <img src="__IMG_SAMPLE_IMAGES__" class="figure-img" style="max-height: 190px;" alt="Sample Preprocessed Signs">
  <div class="figure-caption">Figure 4: Sample preprocessed 32x32 traffic signs representing varied sign morphologies and lighting variations across GTSRB classes.</div>
</div>

<!-- ========================================================================= -->
<!-- STAGE 5: DATA BALANCING & LEAKAGE CONTROL                                 -->
<!-- ========================================================================= -->
<h2>Stage 5: Data Balancing Strategy &amp; Data Leakage Prevention</h2>

<div class="callout callout-warn">
  <div class="callout-title">The Physical Sign Track Grouping Rule (Zero Leakage)</div>
  <p>
    In GTSRB, the training pool consists of video sequences where a vehicle drove past a physical sign, recording roughly 30 successive camera frames. If a student performs a standard random 80/20 split, frame #14 of a physical sign ends up in training while frame #15 ends up in validation. Because adjacent video frames are nearly identical, a simple nearest-neighbor or memorizing tree will achieve artificial 99% accuracy. 
    <strong>RoadEye eliminates this flaw:</strong> We group images by physical track (1,307 tracks). All 30 frames of any given physical sign are assigned strictly to training OR validation using <code>StratifiedGroupKFold</code>.
  </p>
</div>

<h3>Why We Rejected Synthetic Oversampling (SMOTE)</h3>
<p>
  Synthetic Minority Over-sampling Technique (SMOTE) synthesizes new points by linear interpolation between nearest neighbors in feature space. In our project, SMOTE was formally evaluated and rejected for two vital reasons:
</p>
<ol>
  <li><strong>Unphysical Feature Vectors:</strong> HOG descriptors are non-linear, normalized block histograms of oriented gradients. A linear interpolation between two 1,764-dimensional HOG vectors does not represent a valid physical image of a sign.</li>
  <li><strong>Cost-Sensitive Learning is Superior:</strong> Instead of hallucinating synthetic data, we adopted cost-sensitive loss weighting via <code>class_weight='balanced'</code>. This weights the loss for class j inversely proportional to its frequency:
    <div class="math-block">w_j = N / (K * n_j)</div>
    where N is total training samples, K=43 classes, and n_j is the count for class j. Rare signs (e.g. Class 0, n=210) receive 10.7x higher penalization for errors than dominant signs.
  </li>
</ol>

<!-- ========================================================================= -->
<!-- STAGE 6: EXPLORATORY DATA ANALYSIS (EDA)                                  -->
<!-- ========================================================================= -->
<div class="page-break"></div>
<h1>Stage 6: Exploratory Data Analysis (EDA)</h1>

<p>
  In accordance with Section 5 of the Innovative Assignment guidelines, Exploratory Data Analysis must be <em>analytical, not decorative</em>. EDA was conducted to directly inform our feature engineering and algorithmic selection decisions:
</p>

<div class="grid-2">
  <div class="figure-box">
    <img src="__IMG_SIZE_DIST__" class="figure-img" style="max-height: 185px;" alt="Image Size Distribution">
    <div class="figure-caption">Figure 5: Aspect ratio and resolution distribution. The vast majority of signs are approximately square (aspect ratio 1.0) with median size 43x43.</div>
  </div>
  <div class="figure-box">
    <img src="__IMG_BRIGHTNESS__" class="figure-img" style="max-height: 185px;" alt="Brightness and Contrast">
    <div class="figure-caption">Figure 6: Mean luminance vs RMS contrast. Massive dispersion proves raw pixel brightness is highly unstable and will confuse linear classifiers.</div>
  </div>
</div>

<h2>Key EDA Insights &amp; Technical Consequences</h2>
<ol>
  <li><strong>Luminance Dispersion Mandates Gradient Features:</strong> As demonstrated in Figure 6, images of the exact same sign class range from extreme underexposure (mean pixel value &lt; 30) to heavy solar washout (&gt; 200). A raw-pixel classifier will fail because identical signs have divergent intensity vectors. This directly justifies HOG's gradient-magnitude and block normalization approach.</li>
  <li><strong>Visual Look-Alike Clusters:</strong> Figure 7 illustrates the class-mean representations. Certain sign families share identical circular outlines and differing internal pictograms:
    <ul>
      <li><strong>Speed Limit Cluster:</strong> Classes 0, 1, 2, 3, 4, 5, 7, 8 (Speed limits 20 to 120 km/h). All share red circular borders with white centers; the only discriminating information is the fine central digit glyph.</li>
      <li><strong>Warning Triangle Cluster:</strong> Classes 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31 (triangles with apex pointing up).</li>
      <li><strong>Mandatory Circular Blue Signs:</strong> Classes 33, 34, 35, 36, 37, 38, 39, 40 (directional arrows).</li>
    </ul>
    Because circular speed limits share 85% of their boundary structure, classifiers require high-frequency orientation bins to resolve the internal numbers.
  </li>
</ol>

<div class="figure-box">
  <img src="__IMG_CLASS_MEANS__" class="figure-img" style="max-height: 200px;" alt="Class Mean Images">
  <div class="figure-caption">Figure 7: Pixel-wise mean images across all 43 GTSRB classes, revealing structural templates and cluster similarities.</div>
</div>

<!-- ========================================================================= -->
<!-- STAGE 7: FEATURE ENGINEERING (HOG MATHEMATICS)                           -->
<!-- ========================================================================= -->
<div class="page-break"></div>
<h1>Stage 7: Feature Engineering &amp; The Mathematics of HOG</h1>

<p>
  Histogram of Oriented Gradients (Dalal &amp; Triggs, CVPR 2005) is the core feature representation in RoadEye. HOG captures local edge directions and structural shape while discarding irrelevant global illumination.
</p>

<h2>7.1 Mathematical Derivation of HOG Descriptors</h2>
<p>
  Given an input grayscale image I(x,y) resized to 32x32 pixels:
</p>

<h4>Step 1: Gradient Computation via Central Difference Kernels</h4>
<p>
  The horizontal and vertical gradient components are computed by filtering with 1D masks [-1, 0, 1]:
</p>
<div class="math-block">
  G_x(x,y) = I(x+1, y) - I(x-1, y), &emsp; G_y(x,y) = I(x, y+1) - I(x, y-1)
</div>
<p>
  For each pixel, the gradient magnitude m(x,y) and unsigned orientation angle &theta;(x,y) are:
</p>
<div class="math-block">
  m(x,y) = &radic;(G_x(x,y)&sup2; + G_y(x,y)&sup2;), &emsp; &theta;(x,y) = arctan(G_y(x,y) / G_x(x,y)) mod 180&deg;
</div>

<h4>Step 2: Spatial &amp; Orientation Binning into Cells</h4>
<p>
  The image is tiled into non-overlapping spatial <strong>cells</strong> of size 4x4 pixels. The orientation range [0&deg;, 180&deg;] is partitioned into B = 9 discrete histogram bins (each bin spans 20&deg;). Each pixel inside a cell votes into its corresponding orientation bin, weighted by its gradient magnitude m(x,y) with trilinear interpolation between adjacent bins.
</p>

<h4>Step 3: Block Contrast Normalization (L2-Hysteresis)</h4>
<p>
  To achieve illumination invariance, cells are grouped into overlapping spatial <strong>blocks</strong> of 2x2 cells (8x8 pixels). For each block vector v, L2-Hys normalization is applied:
</p>
<div class="math-block">
  v' = v / &radic;(||v||&sup2; + &epsilon;&sup2;), &emsp; v'' = min(v', 0.2), &emsp; v_final = v'' / &radic;(||v''||&sup2; + &epsilon;&sup2;)
</div>
<p class="formula-expl">
  Capping components at 0.2 suppresses excessive gradient spikes caused by specular glare or direct headlights.
</p>

<h2>7.2 Exact Dimensionality Derivation (1,764 Features)</h2>
<div class="callout callout-info">
  <div class="callout-title">The 1,764 Dimensionality Formula (Must-Know for Viva)</div>
  <p>
    The total length of the HOG feature vector is computed as follows:<br>
    &bull; <strong>Cells per axis:</strong> 32 px / 4 px/cell = 8 cells.<br>
    &bull; <strong>Blocks per axis:</strong> Cells per axis - Cells per block + 1 = 8 - 2 + 1 = 7 blocks.<br>
    &bull; <strong>Total overlapping blocks:</strong> 7 * 7 = 49 blocks.<br>
    &bull; <strong>Features per block:</strong> (2 * 2 cells) * 9 orientation bins = 36 features.<br>
    &bull; <strong>Total Feature Vector Dimension:</strong> 49 blocks * 36 features/block = <strong>1,764 features</strong>.
  </p>
</div>

<div class="figure-box">
  <img src="__IMG_HOG_VIS__" class="figure-img" style="max-height: 190px;" alt="HOG Visualization">
  <div class="figure-caption">Figure 8: Visualization of computed HOG gradient vectors across cells. Note how the outer triangular and circular boundaries generate sharp, unambiguous gradient spikes.</div>
</div>

<!-- ========================================================================= -->
<!-- STAGE 8 & 9: BASELINE & MULTI-MODEL COMPARISON                            -->
<!-- ========================================================================= -->
<div class="page-break"></div>
<h1>Stage 8 &amp; 9: Baseline Model &amp; Multi-Model Comparative Study</h1>

<h2>8.1 The Baseline Reference Model (Raw Pixels + SVM)</h2>
<p>
  In accordance with guideline requirements, an advanced model must be justified by demonstrating tangible superiority over a simple, meaningful baseline. 
</p>
<ul>
  <li><strong>Baseline Architecture:</strong> Flattened raw grayscale pixel intensities (32x32 = 1,024 features) &rarr; <code>StandardScaler</code> &rarr; RBF Support Vector Machine (C=1.0, &gamma;='scale').</li>
  <li><strong>Validation Score:</strong> Accuracy = 71.51%, Macro-F1 = 0.6550.</li>
  <li><strong>Test Score:</strong> Accuracy = 72.27%, Macro-F1 = 0.6664.</li>
  <li><strong>Failure Mode:</strong> The baseline heavily confuses dark speed limit signs with dark warning signs because pixel intensities correlate with ambient lighting rather than edge geometry.</li>
</ul>

<h2>9.1 Multi-Algorithm Evaluation on HOG Features</h2>
<p>
  We implemented and benchmarked three distinct supervised learning algorithms on identical HOG features:
</p>

<h3>1. K-Nearest Neighbors (KNN) &bull; Instance-Based Non-Parametric</h3>
<ul>
  <li><strong>Theory:</strong> Predicts class label via majority vote among the k closest training vectors in Euclidean space:
    <div class="math-block">d(x, x') = &radic;(&sum; (x_i - x'_i)&sup2;)</div>
  </li>
  <li><strong>Strengths:</strong> Zero training time; non-linear decision boundaries.</li>
  <li><strong>Weaknesses:</strong> High memory footprint (must store all 31,367 training vectors); high inference latency (0.39 ms/sample); susceptible to the <em>curse of dimensionality</em> in 1,764-dimensional space.</li>
</ul>

<h3>2. Decision Trees (CART) &bull; Hierarchical Rule Partitioning</h3>
<ul>
  <li><strong>Theory:</strong> Recursively splits feature dimensions by choosing threshold t on feature j that maximizes Gini Impurity reduction:
    <div class="math-block">I_G(p) = 1 - &sum; p_k&sup2;</div>
  </li>
  <li><strong>Strengths:</strong> Extremely fast inference (&lt; 0.01 ms/sample); fully interpretable rule paths.</li>
  <li><strong>Weaknesses:</strong> Axis-aligned orthogonal splits cannot model smooth diagonal hyperplanes in continuous gradient space; heavy over-fitting tendency on continuous HOG histograms.</li>
</ul>

<h3>3. Support Vector Machines (SVM) &bull; Maximum-Margin Optimization</h3>
<ul>
  <li><strong>Theory:</strong> Solves the convex dual quadratic optimization problem to find the optimal hyperplane maximizing the functional margin (2 / ||w||):
    <div class="math-block">max &sum; &alpha;_i - 0.5 * &sum; &alpha;_i &alpha;_j y_i y_j K(x_i, x_j) &emsp; s.t. 0 &le; &alpha;_i &le; C, &sum; &alpha;_i y_i = 0</div>
    Using the Radial Basis Function (RBF) kernel: K(x, x') = exp(-&gamma; ||x - x'||&sup2;).
  </li>
  <li><strong>Multiclass Strategy:</strong> One-vs-One (OvO) voting scheme training 43 * 42 / 2 = 903 individual binary classifiers.</li>
  <li><strong>Strengths:</strong> Global optimum guarantee (convex objective); robust against curse of dimensionality via margin regularization; only support vectors determine the boundary.</li>
</ul>

<div class="grid-2">
  <div class="figure-box">
    <img src="__IMG_BASELINE_CM__" class="figure-img" style="max-height: 175px;" alt="Baseline Confusion Matrix">
    <div class="figure-caption">Figure 9: Baseline Confusion Matrix (Raw Pixels). Note substantial off-diagonal dispersion.</div>
  </div>
  <div class="figure-box">
    <img src="__IMG_SVM_CM__" class="figure-img" style="max-height: 175px;" alt="HOG SVM Confusion Matrix">
    <div class="figure-caption">Figure 10: HOG SVM Confusion Matrix. Clean diagonal concentration confirms superior feature separation.</div>
  </div>
</div>

<!-- ========================================================================= -->
<!-- STAGE 10: HYPERPARAMETER TUNING                                           -->
<!-- ========================================================================= -->
<div class="page-break"></div>
<h1>Stage 10: Hyperparameter Tuning via Stratified Group Cross-Validation</h1>

<p>
  Relying on default library parameters is prohibited under the assignment rubric. We conducted rigorous hyperparameter optimization using <code>GridSearchCV</code> embedded within a scikit-learn <code>Pipeline</code>.
</p>

<h2>10.1 The Validation Protocol</h2>
<ul>
  <li><strong>Cross-Validation Scheme:</strong> 3-Fold <code>StratifiedGroupKFold</code> strictly grouped by physical sign track.</li>
  <li><strong>Scoring Function:</strong> <code>scoring='f1_macro'</code> (treats all 43 classes equally, preventing dominant classes from biasing the parameter search).</li>
  <li><strong>Zero-Leakage Pipeline:</strong> <code>Pipeline([('scaler', StandardScaler()), ('classifier', SVC())])</code> ensures that feature standardization is fitted solely on the 2 training folds and applied transform-only to the validation fold during each iteration.</li>
</ul>

<h2>10.2 Parameter Grids and Optimal Search Results</h2>

<table>
  <thead>
    <tr>
      <th>Model</th>
      <th>Hyperparameter Search Grid</th>
      <th>Best Selected Parameters</th>
      <th>CV Best Macro-F1</th>
    </tr>
  </thead>
  <tbody>
    <tr class="best-row">
      <td><strong>SVM (RBF)</strong></td>
      <td>
        C in {0.1, 1.0, 10.0, 100.0}<br>
        &gamma; in {'scale', 0.001, 0.01}<br>
        Kernel: RBF, Linear
      </td>
      <td><strong>Kernel: RBF, C = 10.0, &gamma; = 'scale'</strong></td>
      <td><strong>0.9356</strong></td>
    </tr>
    <tr>
      <td><strong>KNN</strong></td>
      <td>
        k in {3, 5, 7, 9}<br>
        Weights: 'uniform', 'distance'
      </td>
      <td>k = 7, Weights: 'distance'</td>
      <td>0.8852</td>
    </tr>
    <tr>
      <td><strong>Decision Tree</strong></td>
      <td>
        Criterion: Gini, Entropy<br>
        Max Depth: 10, 15, 20, None<br>
        Min Samples Leaf: 1, 2, 4
      </td>
      <td>Criterion: Entropy, Max Depth: 20, Min Leaf: 1</td>
      <td>0.7430</td>
    </tr>
  </tbody>
</table>

<div class="figure-box">
  <img src="__IMG_GRID_SEARCH__" class="figure-img" style="max-height: 200px;" alt="Grid Search SVM Surface">
  <div class="figure-caption">Figure 11: Cross-validation Macro-F1 surface across SVM hyperparameter combinations. C=10 with &gamma;='scale' yielded the global optimum.</div>
</div>

<!-- ========================================================================= -->
<!-- STAGE 11: FINAL EVALUATION & ERROR ANALYSIS                               -->
<!-- ========================================================================= -->
<div class="page-break"></div>
<h1>Stage 11: Final Evaluation, Error Analysis &amp; Validation</h1>

<h2>11.1 The Single-Shot Final Test Protocol</h2>
<p>
  A critical methodological safeguard in RoadEye is our <strong>single-shot test evaluation lock</strong>. The official GTSRB test set (12,630 images) was quarantined during all EDA, feature engineering, model selection, and hyperparameter tuning. The test set was accessed exactly once after all parameters were frozen. A lock file (results/metrics/.test_evaluated.lock) records the SHA-256 model hash and timestamp to prove that no test snooping or post-hoc threshold tweaking occurred.
</p>

<h2>11.2 Comprehensive Final Test Metrics</h2>
<ul>
  <li><strong>Total Test Samples Evaluated:</strong> 12,630 unseen images across 43 classes.</li>
  <li><strong>Overall Accuracy:</strong> <strong>94.08%</strong> (11,882 / 12,630 correct predictions).</li>
  <li><strong>Macro-Averaged Precision:</strong> <strong>0.9521</strong> (95.21%).</li>
  <li><strong>Macro-Averaged Recall:</strong> <strong>0.9069</strong> (90.69%).</li>
  <li><strong>Macro-Averaged F1 Score:</strong> <strong>0.9256</strong> (92.56%).</li>
  <li><strong>Weighted-Averaged F1 Score:</strong> <strong>0.9401</strong> (94.01%).</li>
  <li><strong>Balanced Accuracy:</strong> <strong>0.9069</strong> (90.69%).</li>
</ul>

<div class="grid-2">
  <div class="figure-box">
    <img src="__IMG_FINAL_CM__" class="figure-img" style="max-height: 200px;" alt="Final Confusion Matrix">
    <div class="figure-caption">Figure 12: Final Test Confusion Matrix for Tuned HOG-SVM on all 12,630 images.</div>
  </div>
  <div class="figure-box">
    <img src="__IMG_FINAL_F1__" class="figure-img" style="max-height: 200px;" alt="Per-Class F1 Score">
    <div class="figure-caption">Figure 13: Per-class F1 distribution. 38 of 43 classes exceed 0.90 F1.</div>
  </div>
</div>

<h2>11.3 Detailed Error Analysis: Where and Why Does the Model Fail?</h2>
<p>
  Analyzing failure cases is a mandatory requirement under Section 7 of the assignment rubric. The model committed 748 classification errors out of 12,630 test images (5.92% error rate). Detailed forensic analysis of the confusion matrix reveals three primary error clusters:
</p>

<ol>
  <li><strong>Speed Limit Glyph Confusion (Intra-Cluster Look-Alikes):</strong>
    The highest frequency errors occurred between Class 1 (Speed limit 30 km/h) and Class 2 (Speed limit 50 km/h), as well as Class 7 (100 km/h) and Class 8 (120 km/h). In low-resolution 32x32 images, a blurred digit '3' and digit '5' produce nearly identical gradient distributions.
  </li>
  <li><strong>Pedestrian Crossing (Class 27) vs General Danger (Class 18):</strong>
    Both signs are red equilateral warning triangles with white interiors and black central pictograms. When images are degraded by rain or motion blur, the walking human silhouette in Class 27 loses high-frequency details and is misclassified as the exclamation mark of Class 18.
  </li>
  <li><strong>Small Resolution Degradation (&lt; 30x30 pixels):</strong>
    Signs captured at distances exceeding 50 meters occupy fewer than 28x28 pixels on the raw camera sensor. Upsampling these small patches via bilinear interpolation creates blurred gradients that lack the orientation definition required for HOG.
  </li>
</ol>

<div class="figure-box">
  <img src="__IMG_MISCLASSIFIED__" class="figure-img" style="max-height: 190px;" alt="Misclassified Examples">
  <div class="figure-caption">Figure 14: Sample misclassified test images showing true vs predicted classes under heavy motion blur and severe shadow occlusion.</div>
</div>

<!-- ========================================================================= -->
<!-- STAGE 12: USE CASE, IMPACT & LIMITATIONS                                  -->
<!-- ========================================================================= -->
<div class="page-break"></div>
<h1>Stage 12: Practical Application, Deployment &amp; Limitations</h1>

<h2>12.1 The Streamlit Web Application</h2>
<p>
  To prove that our solution is operational and usable (as required by Section 1 &amp; Section 8 of the guidelines), we implemented a full-featured web dashboard in app/app.py.
</p>
<ul>
  <li><strong>Real-Time Image Upload:</strong> Users upload arbitrary sign photographs (.png, .jpg, .ppm).</li>
  <li><strong>Identical Transformation Pipeline:</strong> The application automatically invokes the exact same preprocessing routines (bounding crop, grayscale conversion, 32x32 bilinear resizing, 1,764-d HOG extraction, StandardScaler transformation) ensuring zero inference discrepancy.</li>
  <li><strong>Diagnostic Margin Ranking:</strong> The app displays the top-3 predicted classes alongside their raw SVM decision function margins.</li>
  <li><strong>Safety-First Disclaimer:</strong> Clear operational warning that the system is an educational prototype and not safety-certified under ISO 26262 automotive standards.</li>
</ul>

<div class="callout callout-warn">
  <div class="callout-title">Technical Justification: Why We Do NOT Display "Probabilities"</div>
  <p>
    Many student projects call <code>predict_proba()</code> on SVMs and display a percentage (e.g. <em>"98.4% confident"</em>). <strong>In our viva, we explain why this is scientifically unsound:</strong> Standard SVMs are maximum-margin classifiers that compute uncalibrated geometric distances to hyperplanes, not true posterior probabilities. Generating probabilities requires Platt scaling (fitting an external logistic regression on top of decision values via cross-validation). Showing a synthetic percentage without rigorous calibration is dangerously misleading in safety-critical robotics. We intentionally report raw decision margins and rank order.
  </p>
</div>

<h2>12.2 Real-World Limitations</h2>
<ol>
  <li><strong>Classification Only (No Object Detection):</strong> The model requires a pre-cropped bounding box of a sign. It does not perform spatial localization or bounding box regression (e.g., YOLO / Faster R-CNN).</li>
  <li><strong>Domain Shift (German vs International Signs):</strong> The model is trained exclusively on German traffic signs. While the Vienna Convention standardizes European signage, US signs (MUTCD) use yellow diamond warning signs and rectangular speed limits that violate these feature priors.</li>
  <li><strong>Color Information Discarded:</strong> Converting to grayscale removes color discriminators (e.g., red circular stop signs vs blue circular mandatory signs).</li>
  <li><strong>Non-Invariance to In-Plane Rotation:</strong> HOG orientation bins [0&deg;, 180&deg;] are sensitive to severe vehicle roll angle or tilted signs.</li>
</ol>

<h2>12.3 Future Scope &amp; Engineering Roadmap</h2>
<ul>
  <li><strong>Stage 1 Integration:</strong> Prepend an edge-based or classical Haar Cascade / MSER detector to isolate sign candidate regions from full windshield dashcam video.</li>
  <li><strong>Color-HOG Concatenation:</strong> Extract HOG independently across Hue-Saturation-Value (HSV) or Lab color channels.</li>
  <li><strong>Dimensionality Reduction:</strong> Apply Principal Component Analysis (PCA) to compress 1,764 features down to 256 components to accelerate inference.</li>
  <li><strong>Deep Learning Comparative Study:</strong> Train a compact CNN (e.g., MobileNetV3) outside the syllabus to establish the theoretical upper bound on accuracy.</li>
</ul>

<!-- ========================================================================= -->
<!-- LIVE DEMONSTRATION WALKTHROUGH FOR THE PROFESSOR                         -->
<!-- ========================================================================= -->
<div class="page-break"></div>
<h1>Live Demonstration Guide: Presenting to Your Professor</h1>

<p>
  When you present RoadEye to your professor, follow this exact step-by-step demonstration sequence to demonstrate total command over the codebase, reproducibility, and deployment:
</p>

<h2>Step 1: Open Terminal &amp; Run Synthetic Verification (5 Seconds)</h2>
<p>
  Prove that the codebase is completely tested and functioning without needing to wait for huge data downloads:
</p>
<div class="code-snip">
python3 -m pytest tests -v
</div>
<p>
  <strong>What to tell the professor:</strong> <em>"Professor, before demonstrating the UI, I'm running our automated test suite. We built a synthetic GTSRB generator in <code>tests/synthetic.py</code> that validates our data loaders, preprocessing, HOG feature extraction, and pipeline orchestration in under 5 seconds."</em>
</p>

<h2>Step 2: Launch the Interactive Streamlit Web App</h2>
<p>
  Launch the web demo locally:
</p>
<div class="code-snip">
streamlit run app/app.py
</div>
<p>
  The browser will automatically open at <code>http://localhost:8501</code>.
</p>

<h2>Step 3: Upload a Sample Sign &amp; Explain the Visuals</h2>
<ol>
  <li>In the file uploader, navigate to <code>data/raw/GTSRB/Final_Test/Images/</code> (or use any traffic sign image).</li>
  <li>Select an image such as a Stop sign or Speed Limit sign.</li>
  <li><strong>Point to the UI panels and explain:</strong>
    <ul>
      <li><strong>Original vs Processed:</strong> Show how the app converts the uploaded image to 32x32 grayscale.</li>
      <li><strong>HOG Feature Histogram:</strong> Show the 1,764-dimensional feature vector visualization.</li>
      <li><strong>Prediction &amp; Top-3 Margins:</strong> Show the predicted class and the top-3 decision scores. Explain: <em>"Notice we display geometric margin scores rather than uncalibrated probabilities, adhering to safety best practices."</em></li>
    </ul>
  </li>
</ol>

<h2>Step 4: Demonstrate the Pre-Trained Metrics and Reproducibility</h2>
<p>
  Open the <code>results/</code> directory in VS Code or Terminal to prove all experimental figures and metrics were generated by code, not hardcoded:
</p>
<div class="code-snip">
ls -lh results/metrics/
cat results/model_comparison/model_comparison.csv
</div>
<p>
  <strong>What to say:</strong> <em>"Every number in our report was logged directly to JSON and CSV files by our automated pipeline runner. The test set was locked with an SHA-256 hash to ensure zero data leakage."</em>
</p>

<!-- ========================================================================= -->
<!-- VIVA DEFENSE MASTER COMPENDIUM (TOP 35 QUESTIONS & ANSWERS)              -->
<!-- ========================================================================= -->
<div class="page-break"></div>
<h1>The Professor's Viva Defense Compendium: 35 Core Questions</h1>

<p>
  Below are the top 35 theoretical, architectural, and mathematical questions professors ask during machine learning viva examinations, paired with bulletproof model answers.
</p>

<h3>Category A: HOG Feature Engineering Mathematics</h3>

<div class="viva-item">
  <div class="viva-q">What is HOG and why did you choose it over raw pixels or SIFT/SURF?</div>
  <div class="viva-a">
    Histogram of Oriented Gradients (HOG) tiles an image into cells, computes gradient orientations, and constructs local histograms normalized across overlapping blocks. Traffic signs are defined by structural outlines and high-contrast pictograms (edges). HOG captures edge orientation while discarding global illumination. Raw pixels fail under lighting shifts; SIFT/SURF are sparse interest-point detectors designed for matching, whereas HOG provides a dense structural descriptor ideal for dense classification.
  </div>
</div>

<div class="viva-item">
  <div class="viva-q">How exactly did you arrive at 1,764 features? Show the math.</div>
  <div class="viva-a">
    Image is 32x32 pixels. Cells are 4x4 pixels &rarr; 8x8 cells. Blocks are 2x2 cells with a stride of 1 cell &rarr; (8 - 2 + 1) = 7 blocks horizontally and vertically &rarr; 49 total blocks. Each block has 4 cells, and each cell has 9 orientation bins &rarr; 4 * 9 = 36 values per block. 49 blocks * 36 values/block = 1,764 features.
  </div>
</div>

<div class="viva-item">
  <div class="viva-q">What is the purpose of L2-Hysteresis block normalization in HOG?</div>
  <div class="viva-a">
    Block normalization divides gradient vectors by their Euclidean norm over an 8x8 pixel region, making features invariant to shadows and illumination shifts. The "hysteresis" step clips normalized values at 0.2 and re-normalizes, preventing intense specular highlights or glare from dominating the feature vector.
  </div>
</div>

<div class="viva-item">
  <div class="viva-q">Why use unsigned gradients [0&deg;, 180&deg;] instead of signed [0&deg;, 360&deg;]?</div>
  <div class="viva-a">
    Unsigned gradients treat a light-to-dark edge identically to a dark-to-light edge. For traffic signs, this ensures that a black symbol on a white background produces the same gradient orientation as a white symbol on a blue background, increasing model generalization across sign color schemes.
  </div>
</div>

<h3>Category B: Support Vector Machine (SVM) Theory</h3>

<div class="viva-item">
  <div class="viva-q">How does an SVM find the optimal decision boundary?</div>
  <div class="viva-a">
    SVM solves a convex quadratic programming problem to find the separating hyperplane that maximizes the geometric margin (2 / ||w||) between the closest training points of opposing classes (support vectors). Only support vectors influence the boundary position; non-support vector points can be added or removed without altering the solution.
  </div>
</div>

<div class="viva-item">
  <div class="viva-q">What do hyperparameters C and gamma control in an RBF SVM?</div>
  <div class="viva-a">
    C is the regularization parameter trading margin width against training error penalty. Large C penalizes misclassifications heavily (narrow margin, risks over-fitting); small C tolerates errors for a wider, smoother margin (risks under-fitting). &gamma; defines the radius of influence of the RBF kernel. Large &gamma; means support vectors have localized influence (complex, wavy boundaries, over-fitting); small &gamma; means broad influence (smoother, linear-like boundary).
  </div>
</div>

<div class="viva-item">
  <div class="viva-q">How does your SVM handle 43 classes when it is inherently a binary classifier?</div>
  <div class="viva-a">
    We use the One-vs-One (OvO) strategy implemented by scikit-learn's SVC (LIBSVM). It trains a binary SVM for every pair of classes: 43 * 42 / 2 = 903 individual binary models. At test time, each binary classifier casts a vote, and the class receiving the majority of votes is assigned as the final prediction.
  </div>
</div>

<div class="viva-item">
  <div class="viva-q">Why is feature scaling (StandardScaler) strictly mandatory before SVM?</div>
  <div class="viva-a">
    The RBF kernel computes Euclidean distances between vectors: ||x - x'||&sup2;. If one feature has a range of [0, 1000] and another [0, 1], the larger feature will completely dominate the distance metric regardless of its actual predictive power. StandardScaler scales all 1,764 features to mean 0 and variance 1.
  </div>
</div>

<div class="page-break"></div>
<h3>Category C: Data Leakage, Validation &amp; Imbalance</h3>

<div class="viva-item">
  <div class="viva-q">What is data leakage and how did your project specifically prevent it?</div>
  <div class="viva-a">
    Data leakage occurs when information from the validation or test set inadvertently influences model training. We prevented it through three mechanisms: (1) <strong>Group Splitting:</strong> GTSRB has ~30 consecutive frames per physical sign; we split strictly by physical track ID so identical signs never appear in both train and validation. (2) <strong>In-Fold Scaling:</strong> StandardScaler was fit strictly inside cross-validation training folds via scikit-learn Pipelines. (3) <strong>Quarantined Test Set:</strong> The 12,630 test set was held out and evaluated exactly once after all tuning was finished.
  </div>
</div>

<div class="viva-item">
  <div class="viva-q">Why did you use Macro-F1 instead of standard Accuracy?</div>
  <div class="viva-a">
    GTSRB has severe class imbalance (ratio 10.71:1, ranging from 2,250 samples down to 210). Standard accuracy treats all samples equally, meaning a naive model that predicts only the majority classes can achieve 85% accuracy while completely failing on rare, life-critical signs (like Dangerous Curve or Ice/Snow). Macro-F1 calculates the F1-score for each class independently and averages them equally, exposing poor minority performance.
  </div>
</div>

<div class="viva-item">
  <div class="viva-q">Why did you reject SMOTE for data balancing?</div>
  <div class="viva-a">
    SMOTE synthesizes minority points by linear interpolation in feature space. In a 1,764-dimensional HOG space, linearly interpolating between two block-normalized histograms produces an unphysical vector that violates HOG normalization constraints and does not correspond to any valid sign image. Instead, we used cost-sensitive learning (<code>class_weight='balanced'</code>), which directly adjusts the SVM penalty parameter C_j proportionally to class rarity.
  </div>
</div>

<div class="viva-item">
  <div class="viva-q">Why is StratifiedGroupKFold necessary over standard StratifiedKFold?</div>
  <div class="viva-a">
    Standard StratifiedKFold maintains class proportions but randomly shuffles samples. In GTSRB, consecutive video frames of the same physical sign would end up in both training and validation folds, causing massive leakage and falsely inflated validation metrics. <code>StratifiedGroupKFold</code> simultaneously preserves class balance while guaranteeing that all frames belonging to the same physical sign track stay inside the same fold.
  </div>
</div>

<h3>Category D: Comparative Algorithms (KNN &amp; Decision Trees)</h3>

<div class="viva-item">
  <div class="viva-q">Why did Decision Trees perform the worst (73.70% accuracy) on HOG features?</div>
  <div class="viva-a">
    Decision trees perform axis-aligned, orthogonal threshold splits on single features (x_j &le; &theta;). HOG features are continuous, highly correlated gradient distributions where complex diagonal boundaries in hyperspace are required to separate classes. To approximate smooth diagonal hyperplanes, a decision tree must construct an enormous staircase of splits, resulting in severe over-fitting and memorization of training noise.
  </div>
</div>

<div class="viva-item">
  <div class="viva-q">What is the "Curse of Dimensionality" and how does it affect KNN in this project?</div>
  <div class="viva-a">
    In 1,764-dimensional space, the volume of the space grows exponentially, causing all data points to become nearly equidistant from one another. This degrades the discriminative power of Euclidean distance. Furthermore, KNN requires storing all 31,367 training vectors and computing 31,367 Euclidean distances for every single test image, resulting in high inference latency (0.39 ms vs &lt; 0.01 ms for trees).
  </div>
</div>

<div class="viva-item">
  <div class="viva-q">Does a Decision Tree require feature scaling? Why did you scale it anyway?</div>
  <div class="viva-a">
    No. Decision tree split criteria (Gini Impurity, Information Gain) depend purely on the monotonic ordering of values along an axis, not their numerical scale. However, to maintain an identical, fair, and scientifically valid benchmark across all models, all classifiers were trained on the exact same scaled feature matrix.
  </div>
</div>

<h3>Category E: Practical Deployment &amp; Limitations</h3>

<div class="viva-item">
  <div class="viva-q">Why does your web app show decision margins instead of probabilities?</div>
  <div class="viva-a">
    Standard SVMs compute geometric signed distances to the hyperplane (w^T &phi;(x) + b). Generating probabilities requires Platt scaling (fitting a logistic sigmoid via cross-validation). In safety-critical automotive applications, uncalibrated probabilities create false overconfidence. Displaying raw margin scores and ordinal ranks is more scientifically honest and avoids misleading the user.
  </div>
</div>

<div class="viva-item">
  <div class="viva-q">What are the primary limitations of your system in real-world deployment?</div>
  <div class="viva-a">
    (1) It performs classification only, assuming an external sensor or algorithm has already localized and cropped the sign. (2) It was trained only on German signs; American or Asian signs with different shapes/colors will fail. (3) Grayscale conversion discards color cues. (4) Bilinear downsampling to 32x32 pixels blurs fine internal glyphs on distant signs. (5) HOG is not invariant to severe camera rotation or perspective tilt.
  </div>
</div>

<div class="viva-item">
  <div class="viva-q">How would you extend this project in future work?</div>
  <div class="viva-a">
    (1) Add a YOLO or Haar-cascade object detector for end-to-end full-frame video processing. (2) Extract multi-channel color HOG (HSV or Lab color spaces). (3) Apply PCA dimensionality reduction to compress 1,764 features to 256 for faster inference. (4) Compare against an optimized MobileNetV3 deep learning architecture.
  </div>
</div>

<!-- ========================================================================= -->
<!-- ACADEMIC INTEGRITY & GENERATIVE AI STATEMENT                              -->
<!-- ========================================================================= -->
<div class="page-break"></div>
<h1>Academic Integrity, AI Usage Disclosure &amp; References</h1>

<h2>Academic Integrity Compliance (Section 9 of Guidelines)</h2>
<p>
  This project strictly complies with the academic integrity guidelines mandated by Nirma University for the Machine Learning Innovative Assignment:
</p>
<ul>
  <li><strong>Empirical Reproducibility:</strong> No results, metrics, confusion matrices, or validation scores were hand-typed or fabricated. Every reported metric was generated deterministically by the modular training pipeline (scripts/run_pipeline.py) using fixed random seed 42.</li>
  <li><strong>No Pre-Trained External Models:</strong> No pre-trained weights, transfer learning checkpoints, or external classification APIs were used. All baseline, KNN, Decision Tree, and SVM models were trained from scratch on the GTSRB training split.</li>
  <li><strong>Use of Generative AI Assistance:</strong> In compliance with Section 9 of the guidelines, we transparently disclose that AI-assisted tools were utilized for brainstorming, script boilerplate generation, and formatting assistance. All underlying mathematical principles, pipeline architectures, experimental designs, and viva defense explanations have been independently verified and understood by the student presenter.</li>
</ul>

<h2>Formal Academic Citations</h2>
<ol>
  <li><strong>GTSRB Benchmark Dataset:</strong><br>
    J. Stallkamp, M. Schlipsing, J. Salmen, and C. Igel. <em>"Man vs. computer: Benchmarking machine learning algorithms for traffic sign recognition."</em> Neural Networks, Special Issue: Autonomous Learning, Vol. 32, pp. 323&ndash;332, 2012.
  </li>
  <li><strong>Histogram of Oriented Gradients (HOG):</strong><br>
    N. Dalal and B. Triggs. <em>"Histograms of oriented gradients for human detection."</em> IEEE Computer Society Conference on Computer Vision and Pattern Recognition (CVPR), Vol. 1, pp. 886&ndash;893, 2005.
  </li>
  <li><strong>Support Vector Machines (LIBSVM):</strong><br>
    C.-C. Chang and C.-J. Lin. <em>"LIBSVM: A library for support vector machines."</em> ACM Transactions on Intelligent Systems and Technology, Vol. 2, Issue 3, Article 27, 2011.
  </li>
  <li><strong>scikit-learn Machine Learning Framework:</strong><br>
    F. Pedregosa et al. <em>"Scikit-learn: Machine Learning in Python."</em> Journal of Machine Learning Research (JMLR), Vol. 12, pp. 2825&ndash;2830, 2011.
  </li>
</ol>

</body>
</html>
"""

    for placeholder, data_uri in img_map.items():
        template = template.replace(placeholder, data_uri)

    return template

def main():
    print("Building comprehensive HTML document...")
    html_content = build_html()
    with open(HTML_OUT, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"Wrote HTML to {HTML_OUT} ({len(html_content):,} bytes)")

    print("Rendering PDF via headless Chrome...")
    chrome_bin = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
    cmd = [
        chrome_bin,
        "--headless",
        "--disable-gpu",
        "--no-pdf-header-footer",
        f"--print-to-pdf={PDF_OUT}",
        HTML_OUT
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"Error rendering PDF: {res.stderr}")
        sys.exit(1)

    print(f"PDF successfully created at: {PDF_OUT}")

    # Count pages
    with open(PDF_OUT, "rb") as f:
        data = f.read()
    page_count = len(re.findall(rb'/Type\s*/Page[^s]', data))
    print(f"Total Page Count: {page_count} pages")

if __name__ == "__main__":
    main()
