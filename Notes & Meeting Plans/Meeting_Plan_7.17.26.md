# Meeting Plan – 3:00 PM

## Agenda

### 1. Explainability Findings

* Review the explainability work completed for the AASIST V3 model.
* Discuss the evaluation of both SHAP and LIME.
* Present the key findings, including:

  * Why LIME was not applicable to the AASIST V3 architecture.
  * How SHAP was used to identify the most influential embedding dimensions.
  * The correlation analysis between learned embeddings and interpretable audio features.
  * Conclusions regarding the latent feature representations learned by the model.

### 2. Review of Visualizations and Explainability Code

* Walk through the visualizations generated during the explainability analysis.
* Review the explainability functions and implementation.
* Team members are encouraged to review the GitHub repository before or after the meeting, particularly the **`explainability/SHAP_vs_LIME.md`** document, which summarizes:

  * Project motivation
  * Explainability methodology
  * Results and interpretation
  * Design decisions and limitations

### 3. AWS Deployment Demonstration

* Demonstrate the current deployment of the spoof detection model on AWS.
* Discuss the current functionality and user workflow.
* Note that the deployed application currently **does not include the explainability features**, as those components are still being finalized and validated.
* **a video version of the demonstration will be available within this github repo as well**

### 4. Model Strategy Discussion

* Discuss whether to proceed with **AASIST V3 as the standalone production model** or to develop an **ensemble of AASIST V3 and Wav2Vec2**.
* Review the potential benefits and tradeoffs of each approach, including:

  * Predictive performance
  * Model complexity
  * Inference time and deployment considerations
  * Explainability implications
* Determine the preferred direction for the remainder of the project.

### 5. Next Steps

* Gather feedback on the explainability approach and results.
* Identify any remaining work items and priorities.
* Discuss the timeline for integrating explainability into the deployed application.
