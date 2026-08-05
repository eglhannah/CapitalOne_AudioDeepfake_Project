# Capstone Team Meeting Plan

**Date:** July 24, 2026  
**Time:** 3:00 PM

## Attendees

- Chase
- Arnav
- Hannah

**Absent:**
- Mohini

---

## Agenda

### 1. SHAP Explainability Update (Hannah)

- Review recent progress on SHAP explainability for the AASIST v3 model.
- Demonstrate the new function that generates SHAP values for an input audio file.
- Discuss current limitations:
  - Due to the `SELU(inplace=True)` component within the AASIST v3 architecture, it is not currently possible to attribute predictions to specific portions or timestamps of the input audio.
  - As a result, the implementation provides SHAP values for the model representation but cannot produce temporal explanations showing which segments of the audio contributed most to a spoof prediction.

### 2. Deployed Application Updates (Chase)

- Demonstrate the requested updates made to the deployed application.
- Review implemented changes.
- Gather feedback from the team and identify any remaining enhancements or issues.

### 3. Final Report Updates (Hannah on Mohini's Behalf)

- Written final report progress
- Powerpoint progress

### 4. Discussion & Next Steps

- Review outstanding project tasks.
- Identify any blockers or technical challenges.
- Assign action items and priorities for the coming week.
