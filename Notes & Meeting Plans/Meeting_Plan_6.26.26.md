# Project Planning Update

## Overview

The project team met to review current progress, resolve outstanding technical issues, and align on priorities for the coming week. Overall, the team is making strong progress across all workstreams and is confident in the current direction of the project. Key technical milestones have been completed, and remaining tasks are focused primarily on infrastructure setup and integration rather than core development. The team also submitted their Milestone 1 document on Friday June 19th.

## Progress Updates

### Model Development & Explainability

* Troubleshooting related to model loading has been completed, allowing the explainability workflow to be applied consistently across different models.
* The explainability functions are being finalized and will be uploaded to the shared repository before the 3pm team meeting.

### Model Evaluation

* The prediction pipeline has been validated after identifying that only three of the four test data partitions were previously being evaluated.
* Updated prediction results have been generated and uploaded, ensuring future evaluations are performed on the complete test dataset.

### Infrastructure

* The primary blocker discussed during the meeting is AWS account access and free-tier eligibility.
* The team has created an AWS account and configured this account for group use, although there are no free-tier credits available for this account
* The team seeks to discuss with Capital One if this free tier allows the project to deploy as desired

## Action Items

| Team Member          | Action Item                                                                                                                                                                            |
| -------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Chase**            | Create a new AWS account using new credentials and provide a status update to the team.                                                                                                |
| **Hannah**           | Finalize code for explainability and feature extraction, including reasoning and justification for decisions made |
| **All Team Members** | Upload model paths to hugging face and send Hannah the links so she is able to load models for feature extraction and explainability analysis                                                                 |

## Upcoming Milestones

* Complete AWS environment setup.
* Merge explainability functions into the shared repository.
* Continue end-to-end model validation using the complete testing dataset.
* Discuss availability for Capital One visit in Richmond on **Tuesday, July 7**, pending final confirmation.

## Overall Status

The team is in a strong position moving forward. Core technical work is progressing as planned, recent validation efforts have increased confidence in the evaluation pipeline, and the remaining work is centered on infrastructure configuration and final integration. The planning meeting confirmed that the project remains on schedule, with team members confident in the quality of their work and the next steps leading into the upcoming development sprint.
