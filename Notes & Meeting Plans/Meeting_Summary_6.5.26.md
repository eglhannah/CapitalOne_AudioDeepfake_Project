## **Quick recap**

The team met to discuss progress on their voice authentication/spoofing detection project, with Mustufa reviewing Arnav's model results and recommending additional error rate reduction techniques including robust techniques and one-class metrics to bring the error rate from 32% down to 10-15%. Hannah demonstrated SHAP integration for feature explainability, showing visualizations of audio features like jitter spectrum and MFCC variants, while Mohini reported her current EER at 12.6% and mentioned working on a parallel Wave2Vec model approach. The team discussed infrastructure planning for AWS deployment, with Chase agreeing to start working on the infrastructure while Mohini and Arnav focus on their model development, and Mustufa emphasized the importance of documenting the process and decision-making for both the explainability techniques and model comparisons.

## **Next steps**

#### **Chase**

Begin work on AWS infrastructure setup in parallel with model development, ensuring compatibility with finalized model architectures and packages

#### **Hannah**

-   Compare LIME and SHAP explainability techniques for feature visualization, document the comparison, and prepare recommendations
-   Clean up and finalize SHAP integration and feature visualization notebooks, align with both Wave2Vec and Arnav's models, and upload to GitHub
-   Update documentation to include definitions, measurement details, and value ranges for each audio feature used in explainability visualizations
-   Fix Mustufa's GitHub access by removing and re-adding the correct ID to resolve capitalization/login issues
-   Align SHAP integration with Arnav's model as access and computing resources allow

#### **Mehul**

-   Offer troubleshooting/working sessions to the team as needed for infrastructure or implementation issues

#### **Mohini**

-   Research and apply 1-2 EER reduction techniques (e.g., robust, row boost, one class metric) to the Wave2Vec model, document results, and present findings at next week's meeting
-   Present documentation and results of Wave2Vec model at next week's meeting

#### **Mustufa**

-   Review the AWS architecture document shared by Arnab/Chase and provide any inputs or recommendations Collaboration
-   All (Hannah, Mohini, Chase): Finalize decision log documenting which model(s), features, and explainability techniques are selected, with justifications
-   Hannah/Mohini/Chase: Ensure model implementation can handle required audio file types as found in datasets A6/A7 and additional formats for demo purposes
-   All: Measure and document latency profiling across sliding windows (1s, 5s, 10s) for utterance detection as part of final demo preparation
-   Mustufa/Mehul: Plan for a possible team visit to Capital One site and consider arranging a 30-minute demo in front of leadership at project conclusion

## **Summary**

-   The team discussed technical issues with Gemini software that was causing unexpected behavior in cell calculations. They confirmed that Daniel would be joining the meeting, though Ari and Arnav were unable to attend. Mustufa reported reviewing a PDF from Arnav that contained degradation matrix data from 2019 onwards, and mentioned that additional work was needed including at least one more data run, as the current error rate was high at 32%.

**Data Error Reduction Strategies** \* Mustufa discussed strategies to reduce error rates, targeting between 10 to 15%, and recommended implementing at least one standard technique to assess its impact. He shared a PDF analysis comparing clean data sets with real-world data, noting higher error rates in the latter. Mustufa planned to walk through the PDF recommendations to address the question of how to lower these error rates.

**EER Reduction Documentation Strategy** \* Mustufa outlined a documentation strategy for selecting and implementing techniques to reduce EER numbers, including researching multiple options and documenting the decision to use specific techniques like SHAP integration for explainability. The team discussed moving into an implementation phase where they would measure latency across sliding windows for different utterance durations. Hannah expressed confidence in proceeding with the integration of new techniques, noting they now have a good baseline for data preprocessing and optimization, and mentioned plans to document decision-making processes in GitHub.

**EER Reduction and Model Development** \* Mohini reported her current EER at 12.6% and mentioned she is working on documenting her Wave2Vec model process for presentation at next week's meeting. Mustufa recommended implementing additional techniques like row boost and one class metric to further reduce EER, suggesting they select the approach with the lower error rate for implementation. The team discussed infrastructure setup, with Chase offering to begin working on AWS implementation next week while the model development continues, and Hannah noted they are waiting to finalize model structures before setting up hosting.

**AWS Architecture and System Transitions** \* Mustufa discussed reviewing the AWS architecture document sent by Arnab and recommended not waiting for additional weeks to proceed with implementation. Hannah addressed GitHub access issues for Mustufa and mentioned plans to demo SHAP integration. The team discussed limitations with their current computing system, Ravana, and their transition to using Colab due to allocation issues, with plans to seek additional allocations through Daniel Graham.

**Audio Spoofing Model Explainability** \* The team discussed feature weightage explainability for their audio spoofing detection model, with Hannah demonstrating visualizations of five key audio features including jitter spectrum and MFCC variants. Mustufa suggested applying explainability techniques to both Mohini's and Arnav's models to make an informed decision about which approach to implement, rather than selecting just one model at this stage. The team agreed to focus on deploying a single model initially as a priority, with potential for a backup model implementation if time allows, and discussed the importance of explainability for multiple stakeholders including ML engineers, regulatory compliance, and ongoing model monitoring.