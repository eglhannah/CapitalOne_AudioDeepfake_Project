# Capital One Capstone Project Meeting Agenda

**Date:** June 12, 2026
**Time:** 3:00 PM ET
**Project:** Deepfake Audio Detection and Explainability Framework
**Attendees:**

* Capital One Stakeholders
* Hannah
* Mohini
* Arnav Jain
* Chase

---

# Meeting Objectives

The purpose of today's meeting is to:

* Provide updates on model development and evaluation efforts.
* Review recent improvements to the AASIST and Wave2Vec pipelines.
* Compare explainability analyses using both SHAP and LIME.
* Discuss infrastructure, deployment planning, and model management.
* Review progress toward upcoming Capstone deliverables.
* Gather stakeholder feedback on next-phase priorities.

---

# Agenda

## 1. Project Status Overview

### Team Progress Since Last Meeting

* Summary of completed work and ongoing initiatives.
* Review of stakeholder recommendations from prior meetings.
* Discussion of upcoming project milestones and deliverables.

---

## 2. Model Development Updates

### AASIST Model Progress

**Presenter: Arnav**

Review of recent enhancements and evaluation efforts, including:

* Implementation of stakeholder-recommended improvements.
* Cross-domain evaluation results.
* RawBoost augmentation experiments.
* Latency profiling and operational considerations.
* Performance degradation analysis across increasingly challenging datasets.
* Results documented in the updated AASIST v2 writeup.

### Discussion Topics

* Generalization performance across datasets.
* Model robustness across attack types.
* Cross-domain testing outcomes.
* Performance on unseen deepfake audio samples.
* Practical deployment considerations.

---

## 3. Model Variant Comparison

**Presenter: Mohini**

Review of ongoing model training and comparison efforts.

Topics include:

* Current training progress.
* Evaluation of model variants.
* Impact of clip-length modifications.
* Performance implications of boosting techniques.
* Comparative analysis across architectures and configurations.
* Review of stakeholder-requested model improvements.

### Planned Deliverable

A comprehensive comparison of all model variants, including performance trade-offs, strengths, limitations, and recommendations for final model selection.

---

## 4. Explainability and Model Interpretation

**Presenter: Hannah**

Review of explainability initiatives and model interpretation efforts.

### SHAP Analysis

* Development and implementation of an enhanced SHAP workflow for model interpretation.
* Feature attribution analysis across model predictions.
* Identification of influential features and prediction drivers.
* Preliminary findings regarding model behavior.

### LIME Analysis

* Successful implementation of Local Interpretable Model-Agnostic Explanations (LIME).
* Generation of local explanations for individual predictions.
* Evaluation of model decision-making on representative audio samples.

### SHAP vs. LIME Comparison Demonstration

Hannah will provide a walkthrough comparing both explainability approaches, including:

* Differences in methodology.
* Global versus local interpretability capabilities.
* Computational requirements and scalability considerations.
* Strengths and limitations of each approach.
* Suitability for stakeholder reporting and model transparency requirements.

### Explainability Framework Discussion

* Evaluation criteria for explainability method selection.
* Documentation of methodology trade-offs.
* Reproducibility and transparency considerations.
* Recommendations for explainability reporting moving forward.

---

## 5. Infrastructure and Deployment Planning

**Presenter: Chase**

Update on deployment preparation activities.

Topics include:

* AWS environment setup.
* Resource allocation and account provisioning.
* Model serving architecture considerations.
* Deployment strategy evaluation.

### Discussion Topics

* Single-model versus ensemble deployment approaches.
* Infrastructure requirements.
* Cost and scalability considerations.
* Timeline for deployment testing and demonstration.

---

## 6. Rivanna and Research Computing Updates

The team has successfully resolved previous access and storage challenges associated with model development and experimentation.

### Infrastructure Updates

* Daniel Graham provisioned a dedicated allocation for the team within the Rivanna HPC environment.
* Team members now have reliable access to high-performance computing resources required for training, evaluation, and explainability analyses.
* Previous access and connectivity issues have been resolved.
* The team has successfully verified access to the new allocation and is actively utilizing these resources.

### Collaboration and Model Management

To support reproducibility and collaboration:

* A centralized Hugging Face repository has been established for model storage and version management.
* Model checkpoints, trained weights, and associated artifacts are being maintained within this repository.
* The repository provides a scalable solution for sharing large model files that exceed GitHub storage limitations.
* Model versions will continue to be updated as training, tuning, and evaluation efforts progress.

This infrastructure improves experiment tracking, reproducibility, collaboration, and long-term model management.

---

## 7. Documentation and Reproducibility

Review of documentation improvements completed since the previous meeting.

### Current Efforts

* Enhanced GitHub documentation.
* Detailed meeting summaries and project tracking.
* Experiment tracking and reproducibility practices.
* Documentation of modeling decisions and rationale.
* Comparative evaluation records.
* Explainability methodology documentation.
* Model version management and artifact tracking.

### Objectives

* Maintain transparent development practices.
* Ensure repeatable experimentation.
* Document technical decision-making processes.
* Improve stakeholder visibility into project progress and outcomes.

---

# Progress Highlights

## Model Development

Recent efforts have focused on strengthening model robustness and evaluating performance across increasingly realistic attack scenarios.

Key activities include:

* Cross-domain testing.
* RawBoost augmentation experiments.
* Clip-length modification studies.
* Latency profiling.
* Comparative evaluation of model architectures.
* Performance analysis on unseen deepfake datasets.

The team continues to investigate both performance improvements and model generalization characteristics.

---

## Explainability Milestone

The team has now implemented and evaluated two complementary explainability frameworks:

* Enhanced SHAP analysis.
* LIME analysis.

These implementations provide multiple perspectives for understanding model behavior and will inform selection of the final explainability methodology incorporated into project deliverables and stakeholder reporting.

---

## Research Infrastructure Milestone

The team has completed several infrastructure improvements designed to support large-scale experimentation and reproducibility.

Key accomplishments include:

* Dedicated Rivanna allocation established and operational.
* Reliable HPC access confirmed for team members.
* Centralized Hugging Face model repository created.
* Standardized model version management process established.
* Improved sharing of checkpoints and model artifacts.

These improvements strengthen the team's ability to conduct experiments, maintain reproducible workflows, and efficiently manage model assets throughout the remainder of the project.

---

## Deployment Preparation

Infrastructure planning has begun in parallel with model development to support future deployment and demonstration efforts.

Current activities include:

* AWS environment setup.
* Resource planning.
* Access management.
* Evaluation of deployment architectures.
* Preliminary deployment workflow design.

---

## Upcoming Academic Deliverables

### UVA Capstone Progress Report One

**Due:** June 19, 2026

The report will include:

* Project overview and objectives.
* Success criteria.
* Assumptions and limitations.
* Data processing summary.
* Visualizations and results.
* Modeling and analysis summary.
* Future work plan.

The team is drafting an initial version of this report and will finalize for review by the Capital One team by Wednesday (6/17) at 10am.

---

# Discussion and Feedback

The team welcomes stakeholder feedback regarding:

* Model evaluation methodology.
* Explainability framework selection.
* Deployment strategy.
* Documentation and reproducibility practices.
* Priorities for the next phase of development.

---

# Next Steps

## Technical

* Complete model variant comparisons.
* Continue explainability analysis using SHAP and LIME.
* Finalize cross-domain testing evaluations.
* Expand deployment infrastructure setup.
* Continue latency and robustness analysis.
* Refine ensemble and deployment strategies.

## Documentation

* Update technical reports and GitHub documentation.
* Complete Capstone Progress Report One.
* Continue experiment tracking and reproducibility efforts.
* Document explainability methodology selection rationale.

## Stakeholder Engagement

* Incorporate feedback from today's discussion.
* Prioritize analyses requested by Capital One.
* Align future experimentation with deployment objectives and business requirements.

---

**Prepared by:** UVA Deepfake Audio Detection Capstone Team
