# Client Meeting Agenda and Project Update

**Meeting Date:** 6/5/26 @3PM

**Project:** Deepfake Audio Detection Capstone

**Attendees:**

* Daniel Graham (Project Lead)
* Hannah Egl
* Mohini Gupta
* Chase Cha

**Unavailable:**

* Arnav Jain

---

# Meeting Objectives

This meeting will provide project status updates, demonstrate recent technical progress, review explainability efforts, and discuss documentation and reproducibility initiatives requested by project stakeholders.

---

# Agenda

## 1. Introductions and Project Status Overview

* Team attendance and project status update
* Brief review of progress since the previous client meeting
* Confirmation of upcoming milestones and deliverables

---

## 2. Model Performance and Evaluation Updates

### Arnav Jain's Model Progress (Presented by Team)

Review of recent model evaluation results, including:

* Performance on training and validation datasets
* Stratified performance metrics across data subsets
* Performance on previously unseen deepfake datasets
* Analysis of robustness across:

  * Media compression conditions
  * Text-to-Speech (TTS) generated audio
  * Voice Conversion (VC) generated audio
  * Additional attack categories where available

### Key Discussion Topics

* Model generalization performance
* Dataset-specific behavior
* Performance consistency across attack types
* Validation of surprising or unexpectedly strong results

---

## 3. Explainability and Feature Extraction Demonstration

### Presenter: Hannah

Topics include:

* Overview of explainability goals for the deepfake detection system
* SHAP-based explainability demonstration
* Feature extraction workflow and methodology
* Preliminary interpretation of model behavior
* Discussion of explainability approaches under consideration

### Decision Documentation Initiative

The team is developing a structured framework for documenting technical decisions, including:

* Evaluation criteria
* Trade-offs between explainability techniques
* Justification for selected methods
* Reproducibility considerations

This effort is intended to provide greater transparency into model development and decision-making processes.

---

## 4. Rivanna Infrastructure Updates

Discussion topics:

* Current Rivanna environment status
* Model training and experimentation progress
* Resource utilization updates
* Data management and workflow improvements
* Planned infrastructure activities

---

## 5. Documentation, Reproducibility, and Process Improvements

In response to stakeholder feedback, the team has expanded its documentation efforts.

### Current Initiatives

* Enhanced meeting summaries
* Improved GitHub organization and documentation structure
* Documentation of technical decisions and rationale
* Tracking of experimental outcomes and lessons learned
* Documentation of model development workflows
* Reproducibility and repeatability improvements
* Contributor attribution within project documentation

### Ongoing Goals

* Provide clearer visibility into project progress
* Document decision-making processes and trade-offs
* Capture both successful and unsuccessful experimental results
* Maintain comprehensive records of project development activities

---

# Progress Summary Since Previous Meeting

## Assist Model Development

Recent work has focused on evaluating the ASSIST model trained on the ASVspoof 2019 LA dataset.

Highlights include:

* Successful training and validation of the model on clean benchmark data
* Evaluation against increasingly challenging datasets
* Testing under media compression conditions
* Assessment against modern TTS and VC attack scenarios
* Analysis of model robustness across varying audio conditions

The team is continuing to investigate performance characteristics and dataset-specific behavior to ensure findings are fully understood and reproducible.

---

## Explainability Research

The team has begun formal evaluation of explainability approaches for the deepfake detection pipeline.

Current focus areas include:

* SHAP-based explanations
* Feature importance methodologies
* Comparative assessment of explainability techniques
* Development of a decision matrix to support method selection

The objective is to identify an approach that balances interpretability, technical rigor, and stakeholder needs.

---

## Project Management and Communication Enhancements

To improve transparency and collaboration, the team has implemented:

* Pre-meeting accomplishment summaries
* Structured meeting agendas
* Expanded GitHub documentation practices
* Improved tracking of technical decisions
* Standardized reporting of model results and experimental outcomes

---

# Open Discussion

* Client questions and feedback
* Requested analyses or demonstrations
* Upcoming milestones
* Risks, challenges, and mitigation strategies
* Additional stakeholder priorities

---

# Next Steps

### Team Actions

* Continue model evaluation on unseen datasets and attack categories
* Expand explainability analyses and documentation
* Maintain reproducible experimentation workflows
* Enhance technical documentation and decision records
* Prepare materials for upcoming project milestones

### Deliverables Under Development

* Stratified performance reporting
* Explainability methodology documentation
* Decision matrix for explainability selection
* Expanded model evaluation reports
* Reproducibility and workflow documentation

---

**Prepared by:** UVA Deepfake Audio Detection Capstone Team

