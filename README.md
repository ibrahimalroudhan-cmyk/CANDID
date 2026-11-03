# CANDID

![Status](https://img.shields.io/badge/status-research%20prototype-blue) ![Python](https://img.shields.io/badge/python-3.12-blue) ![nnU-Net](https://img.shields.io/badge/nnU--Net-v2-informational) ![Not a medical device](https://img.shields.io/badge/medical%20device-no-lightgrey)

**Canal–tooth ANalysis for Decision-support in Impacted-molar Dentistry**

Research software that measures the distance between the mandibular third molar
and the inferior alveolar canal on CBCT, and withholds the number when it cannot
be trusted. **Not a medical device.**

Project page, worked example, evidence and limitations: **[candidlabs.org](https://candidlabs.org)**

## What is here

A small, deliberately partial slice of the project: volume ingest, mask
extraction and the core geometry, with self-contained checks. The grading,
decision-support and end-to-end layers, trained model weights, validation data
and patient scans are not published.

```bash
cd candid
pip install -r requirements.txt      # Python 3.12 tested
python test_geometry.py
```

## Want more?

Collaboration, validation partnerships, access for research use or just
questions: contact **[Ibrahim@candidlabs.org](mailto:Ibrahim@candidlabs.org)** or see
[candidlabs.org](https://candidlabs.org).

## Licence

Copyright © 2026 Ibrahim Alroudhan. All rights reserved. The code is published for transparency and review; see [LICENSE](LICENSE).
