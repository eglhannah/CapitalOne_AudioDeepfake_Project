# Justification for Selected Audio Features in Deepfake Audio Detection

## Introduction

The increasing sophistication of synthetic speech generation and voice cloning technologies has created significant challenges in distinguishing authentic human speech from artificially generated audio. Deepfake audio systems, particularly those based on neural text-to-speech (TTS) and voice conversion models, can replicate linguistic content with high accuracy while still exhibiting subtle acoustic inconsistencies.

To address this issue, five acoustic features were selected for analysis in this project:

1.  Jitter
2.  Pitch Variability
3.  MFCC Variance
4.  RMS Dynamic Range
5.  Harmonics-to-Noise Ratio (HNR)

These features were chosen because they collectively capture physiological, spectral, temporal, and energetic characteristics of speech that are difficult for synthetic generation systems to reproduce consistently. Together, they provide a comprehensive framework for differentiating authentic human speech from deepfake audio.

------------------------------------------------------------------------

# 1. Jitter

## Definition

Jitter measures small cycle-to-cycle variations in the fundamental frequency (F0) of speech. It reflects the natural instability in vocal fold vibration during phonation.

### Formula

``` math
\text{Jitter} = \frac{1}{N-1} \sum_{i=1}^{N-1} \frac{|T_i - T_{i+1}|}{\bar{T}}
```

Where:

-   (T_i) = duration of a pitch period
-   (\bar{T}) = average pitch period

## Reason for Selection

Human speech production is inherently imperfect due to biological and physiological variability. Real vocal folds do not vibrate with complete periodic consistency, resulting in small fluctuations in pitch periods. Deepfake audio systems, however, frequently generate speech with unnaturally smooth and stable periodicity because neural vocoders and synthesis models optimize for perceptual clarity rather than physiological realism.

By analyzing jitter, the system can detect whether the audio signal contains the subtle irregularities expected in natural human speech. Lower-than-normal jitter values or artificially patterned fluctuations may indicate synthetic generation.

## Importance in Deepfake Detection

Jitter was selected because it provides insight into:

-   Vocal fold realism
-   Biological authenticity
-   Micro-instability in speech production
-   Over-smoothing artifacts common in synthetic speech

This feature is particularly effective for identifying vocoder-generated speech that lacks realistic glottal variability.

------------------------------------------------------------------------

# 2. Pitch Variability

## Definition

Pitch variability measures how the fundamental frequency changes over time throughout an utterance. It reflects the natural prosodic variation present in speech.

### Formula

``` math
\text{Var}(F_0) = \frac{1}{N} \sum_{i=1}^{N}(F_{0,i} - \mu)^2
```

Where:

-   (F\_{0,i}) = pitch value at time (i)
-   (\mu) = mean pitch

## Reason for Selection

Natural human speech contains dynamic pitch movement influenced by emotion, emphasis, stress, sentence structure, and breathing patterns. In contrast, synthetic speech systems often produce flatter or overly controlled intonation contours due to limitations in prosody modeling.

Although modern deepfake systems have improved prosodic generation, they still commonly exhibit:

-   Reduced pitch diversity
-   Excessively smooth transitions
-   Repetitive intonation patterns

Analyzing pitch variability helps identify these inconsistencies.

## Importance in Deepfake Detection

Pitch variability was selected because it captures:

-   Prosodic realism
-   Emotional expressiveness
-   Natural intonation dynamics
-   Temporal pitch fluctuations

This feature is valuable for detecting speech that sounds linguistically correct but lacks authentic human expressiveness.

------------------------------------------------------------------------

# 3. MFCC Variance

## Definition

Mel-Frequency Cepstral Coefficients (MFCCs) are spectral features that represent the short-term power spectrum of speech in a perceptually meaningful manner. MFCC variance measures how these coefficients fluctuate over time.

## Reason for Selection

MFCCs are widely used in speech and speaker recognition because they effectively characterize vocal tract behavior and timbral properties. Natural speech exhibits continuous spectral variation caused by articulation, coarticulation, and anatomical movement.

Synthetic speech often demonstrates:

-   Spectral over-smoothing
-   Reduced articulatory diversity
-   Consistent timbral textures
-   Vocoder artifacts

Measuring MFCC variance enables the detection of these spectral regularities.

## Importance in Deepfake Detection

MFCC variance was selected because it captures:

-   Vocal tract dynamics
-   Spectral richness
-   Articulatory variability
-   Synthetic smoothing artifacts

This feature is particularly useful because many deepfake systems struggle to reproduce the full complexity of natural spectral variation.

------------------------------------------------------------------------

# 4. RMS Dynamic Range

## Definition

Root Mean Square (RMS) energy measures the amplitude or perceived loudness of an audio signal. RMS dynamic range refers to the variation between softer and louder regions of speech.

### Formula

``` math
\text{RMS} = \sqrt{\frac{1}{N} \sum_{i=1}^{N} x_i^2}
```

Where:

-   (x_i) = audio sample value

## Reason for Selection

Human speech naturally contains significant energy fluctuations due to:

-   Emphasis and stress
-   Breathing behavior
-   Consonant-vowel transitions
-   Microphone distance variation
-   Emotional intensity

Synthetic speech systems frequently produce compressed or normalized amplitude patterns in order to maintain clarity and consistency. This often results in speech that sounds unnaturally uniform in loudness.

Analyzing RMS dynamic range allows detection of these artificial energy distributions.

## Importance in Deepfake Detection

RMS dynamic range was selected because it reflects:

-   Natural speech dynamics
-   Loudness variability
-   Energy contour realism
-   Compression artifacts

This feature helps identify synthetic speech that lacks the energetic complexity of authentic human communication.

------------------------------------------------------------------------

# 5. Harmonics-to-Noise Ratio (HNR)

## Definition

Harmonics-to-Noise Ratio (HNR) measures the ratio between periodic harmonic components and aperiodic noise components within speech.

### Formula

``` math
\text{HNR} = 10 \log_{10}\left(\frac{P_{harmonic}}{P_{noise}}\right)
```

Where:

-   (P\_{harmonic}) = harmonic energy
-   (P\_{noise}) = noise energy

## Reason for Selection

Real human voices contain a combination of harmonic structure and physiological noise generated by airflow turbulence, breathiness, and glottal imperfections. Deepfake systems may either:

-   Produce speech that is excessively clean and periodic
-   Introduce artificial noise artifacts through vocoder synthesis

As a result, synthetic speech often exhibits abnormal HNR patterns compared to natural recordings.

## Importance in Deepfake Detection

HNR was selected because it captures:

-   Voice naturalness
-   Breathiness characteristics
-   Harmonic realism
-   Physiological noise behavior

This feature is highly valuable for identifying audio that lacks realistic vocal noise characteristics.

------------------------------------------------------------------------

# Combined Feature Effectiveness

The selected features were intentionally chosen to analyze multiple dimensions of speech behavior rather than relying on a single acoustic indicator. Each feature captures a different aspect of human speech production.

| Feature           | Primary Characteristic              |
|-------------------|-------------------------------------|
| Jitter            | Vocal fold instability              |
| Pitch Variability | Prosodic dynamics                   |
| MFCC Variance     | Spectral and articulatory variation |
| RMS Dynamic Range | Energy fluctuations                 |
| HNR               | Harmonic and noise balance          |

Together, these features provide a robust representation of:

-   Physiological realism
-   Spectral complexity
-   Temporal variability
-   Prosodic naturalness
-   Acoustic authenticity

Because deepfake systems often prioritize perceptual smoothness and intelligibility, they may fail to reproduce the multidimensional irregularities inherent in genuine human speech. The selected features are therefore well-suited for distinguishing synthetic audio from authentic recordings.

------------------------------------------------------------------------

# Conclusion

The five selected features — jitter, pitch variability, MFCC variance, RMS dynamic range, and HNR — were chosen because they collectively represent critical acoustic properties of natural human speech that remain difficult for deepfake systems to replicate perfectly.

These features enable the detection of:

-   Over-smoothing artifacts
-   Unrealistic prosody
-   Reduced spectral diversity
-   Artificial energy consistency
-   Abnormal harmonic structures

By combining physiological, spectral, temporal, and energetic speech characteristics, the selected feature set provides a comprehensive and scientifically grounded approach to deepfake audio detection.