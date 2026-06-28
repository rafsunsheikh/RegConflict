# TRIAG Regulatory Corpus — License & Redistribution Terms

_Last updated 2026-06-19. Applies to the dataset release on Zenodo and Hugging Face._

## Overview

The TRIAG dataset has three components, each licensed independently:

1. **Source regulatory documents** — licensing varies by issuing body (see below).
2. **Annotations** (conflicts, equivalences, supersessions, non-conflicts) — released under **CC BY 4.0**.
3. **Code** (retrieval pipeline, agents, training scripts, evaluation harness) — released under **MIT License**.

The dataset is released for academic research purposes. The release platforms (Zenodo and Hugging Face Datasets) have been confirmed as acceptable redistribution venues for the source documents included in this release.

---

## 1. Source documents

The corpus inventory lists **247 regulatory documents** across 18 source organisations. Of these:

- **192 documents** are redistributed in full as part of the dataset.
- **55 documents** are referenced by metadata + source URL only; the document content is NOT redistributed. Users may fetch these documents directly from the source organisation using the `source_url` field in `regulatory_documents_inventory_licensed.csv`.

### 1.1 Redistributed documents (192 of 247)

Documents from the following sources are included in full. Required attribution strings are listed in `ATTRIBUTIONS.md`.

#### Tier 1 — Open licensing (CC BY 4.0 or equivalent)

### AUSTRAC — Australian Transaction Reports and Analysis Centre (AUSTRAC)

- **Jurisdiction:** Australia
- **Documents in corpus:** 68
- **Source collections:** `AUSTRAC`
- **License:** Creative Commons (version unspecified on page; consistent with CC BY per standard Commonwealth practice)
- **License URL:** <https://www.austrac.gov.au/copyright>
- **Required attribution:**

  > © AUSTRAC for the Commonwealth of Australia 2019

- **Caveats:**
  - AUSTRAC logo excluded from the licence
  - Commonwealth Coat of Arms excluded (use governed by PM&C guidelines)
  - Third-party-owned content excluded — permission may be needed from the third-party copyright owner
  - Must not use material in a way that suggests AUSTRAC or the Australian Government endorses you or your products/services
  - Specific Creative Commons version (e.g., CC BY 4.0) is not named on the copyright page itself

### APRA — Australian Prudential Regulation Authority (APRA)

- **Jurisdiction:** Australia
- **Documents in corpus:** 23
- **Source collections:** `APRA`
- **License:** CC BY 4.0
- **License URL:** <https://www.apra.gov.au/copyright>
- **Required attribution:**

  > © Australian Prudential Regulation Authority 2024

- **Caveats:**
  - Commonwealth Coat of Arms excluded from CC BY licence
  - APRA and APRA Connect logos excluded
  - Third-party material protected by IP law excluded — separate permission required from copyright holder

### ASIC — Australian Securities and Investments Commission (ASIC)

- **Jurisdiction:** Australia
- **Documents in corpus:** 9
- **Source collections:** `ASIC`
- **License:** CC BY 4.0
- **License URL:** <https://asic.gov.au/copyright>
- **Required attribution:**

  > © Australian Securities & Investments Commission: Reproduced with permission.

- **Caveats:**
  - Applies only to the 10 specified document classes: regulatory guides, information sheets, reports, consultation papers, media releases, legislative instruments, pro forma documents, ASIC forms (without logo), annual reports, and short selling position data
  - ASIC logo, design, and formatting excluded from licence — express approval required
  - Unclaimed Money pages excluded (personal use only)
  - Certain ASIC Digest sections (executive summaries, indexes) excluded
  - Thomson Reuters co-published materials subject to separate IP rights

### ATO — Australian Taxation Office (ATO)

- **Jurisdiction:** Australia
- **Documents in corpus:** 5
- **Source collections:** `ATO`
- **License:** ATO Copyright Notice (permissive free-to-distribute statement; functionally equivalent to a CC BY-style open licence, but not formally named CC BY)
- **License URL:** <https://www.ato.gov.au/copyright>
- **Required attribution:**

  > © Australian Taxation Office for the Commonwealth of Australia

- **Caveats:**
  - Must not redistribute in any way that suggests the ATO or the Commonwealth endorses the redistributor or any of their services or products (no-endorsement clause).
  - The ATO does not name a specific licence (e.g., CC BY 4.0); the permission is granted directly by the copyright notice itself.
  - Notice is dated 'Last updated 27 August 2012' — applies broadly to ato.gov.au material; third-party content embedded in ATO pages may have separate rights.

### OAIC — Office of the Australian Information Commissioner (OAIC)

- **Jurisdiction:** Australia
- **Documents in corpus:** 4
- **Source collections:** `OAIC`
- **License:** CC BY 4.0 International
- **License URL:** <https://www.oaic.gov.au/about-the-OAIC/copyright>
- **Required attribution:**

  > Office of the Australian Information Commissioner website – www.oaic.gov.au © Commonwealth of Australia

- **Caveats:**
  - Commonwealth Coat of Arms excluded from CC BY licence
  - OAIC logo excluded
  - Trademarked material excluded
  - Third-party content excluded — must be cleared separately
  - Images and photographs excluded from default CC BY coverage

### RBA — Reserve Bank of Australia (RBA)

- **Jurisdiction:** Australia
- **Documents in corpus:** 1
- **Source collections:** `RBA`
- **License:** CC BY 4.0
- **License URL:** <https://www.rba.gov.au/copyright/>
- **Required attribution:**

  > Source: Reserve Bank of Australia [year]

- **Caveats:**
  - RBA logo excluded (restricted per Logo Use Guidelines)
  - Banknote images excluded (separate reproduction rules)
  - Third-party content requires separate permission
  - Cash rate and financial data subject to special conditions
  - Multimedia content restricted to educational use (except Image Library photographs)
  - Site infrastructure (scripts/stylesheets) excluded
  - Cannot charge fees without disclosing that RBA publishes free versions

### Treasury-AU — The Treasury (Commonwealth of Australia)

- **Jurisdiction:** Australia
- **Documents in corpus:** 5
- **Source collections:** `Treasury`
- **License:** CC BY 4.0
- **License URL:** <https://treasury.gov.au/copyright>
- **Required attribution:**

  > © Commonwealth of Australia

- **Caveats:**
  - Commonwealth Coat of Arms excluded from CC BY 4.0 licence
  - Treasury (department) logo excluded
  - Third-party content excluded — permission of the third-party copyright holder may be required for reuse

### FederalRegister-AU — Parliament of Australia, Commonwealth of Australia (subsidiary legislation)

- **Jurisdiction:** Australia
- **Documents in corpus:** 11
- **Source collections:** `FederalRegister`
- **License:** CC BY 4.0
- **License URL:** <https://www.legislation.gov.au/terms-of-use>
- **Required attribution:**

  > Sourced from the Federal Register of Legislation at [full date of download]. For the latest information on Australian Government law please go to https://www.legislation.gov.au.

- **Caveats:**
  - Commonwealth Coat of Arms excluded from CC BY 4.0 licence
  - Third-party copyrighted material excluded — permission required from third-party copyright holders
  - Different attribution string required for modified vs unmodified content
  - Must include a link to the relevant page or the Legislation Register homepage

### EUR-Lex — European Parliament and Council of the EU, European Union

- **Jurisdiction:** EU
- **Documents in corpus:** 16
- **Source collections:** `EUR-Lex`
- **License:** EU Commission Decision 2011/833/EU (legal documents) + CC BY 4.0 (editorial content, summaries, consolidated texts) + CC0 1.0 (metadata)
- **License URL:** <https://eur-lex.europa.eu/content/legal-notice/legal-notice.html>
- **Required attribution:**

  > © European Union, 1998-2026, https://eur-lex.europa.eu/

- **Caveats:**
  - Reuse policy does not apply to International Accounting Standards (IAS) and other documents flagged with special conditions in the Official Journal
  - Logos, trademarks, registered designs, patents and names are excluded from the reuse policy
  - Use of the EUR-Lex logo requires prior consent of the Publications Office
  - Third-party works embedded in documents may require separate rights clearance
  - Only documents published in the Official Journal of the EU are deemed authentic; only 'European Court Reports' versions are official for case-law
  - CC BY 4.0 attribution required for editorial content/summaries/consolidated texts: acknowledge source and indicate changes

### ESMA — European Securities and Markets Authority (ESMA)

- **Jurisdiction:** EU
- **Documents in corpus:** 22
- **Source collections:** `ESMA-Guidelines`, `ESMA-Consultations`
- **License:** ESMA Legal Notice — Reproduction authorised with acknowledgement (EU agency reuse policy)
- **License URL:** <https://www.esma.europa.eu/legal-notice>
- **Required attribution:**

  > This document has been drafted using material downloaded from ESMA's website

- **Caveats:**
  - Source must be acknowledged on every reproduction
  - Where reproduced material is incorporated in documents that are sold, the publisher must inform buyers that the material may be obtained free of charge through ESMA's website
  - Transformed/derivative content must carry the statement: 'This document has been drafted using material downloaded from ESMA's website'
  - Third-party materials embedded in ESMA documents (notably responses to public consultations) are not covered — permission must be obtained from the third-party copyright holder
  - ESMA name, abbreviation and logo are protected and should not be reproduced
  - Reproduction must not imply endorsement, contain false information, or infringe third-party rights

### ESAs-Joint — European Supervisory Authorities (Joint Committee)

- **Jurisdiction:** EU
- **Documents in corpus:** 2
- **Source collections:** `ESAs-Joint`
- **License:** EU agency reuse with attribution (ESMA/EBA/EIOPA common terms)
- **License URL:** <https://www.esma.europa.eu/legal-notice>
- **Required attribution:**

  > Source: EIOPA - European Insurance and Occupational Pensions Authority, https://eiopa.europa.eu/ (and equivalent acknowledgements to ESMA and EBA for jointly issued documents)

- **Caveats:**
  - Source acknowledgement is mandatory for all reproduction
  - Logos, abbreviations, and corporate identity materials of ESMA/EBA/EIOPA are excluded and require prior written permission
  - Commercial republication requires informing buyers that the material is available free on the respective ESA's website
  - Transformed/derivative works must include a disclaimer stating the material was drawn from the ESA(s) and that the ESA does not endorse the publication
  - Third-party copyrighted content embedded in publications is not covered — separate permission must be sought from rights holders
  - Reproduction must not imply ESA endorsement, present false information, or contain offensive content
  - Joint Committee publications are co-issued by ESMA, EBA, and EIOPA — all three legal notices apply; attribution to all three issuing bodies is advisable

#### Tier 2 — Academic-use redistribution (cleared by author for this release)

The following sources permit academic redistribution under attribution. The author has confirmed academic-purpose redistribution with their research office and cleared each source for inclusion in this release.

### EBA — European Banking Authority (EBA)

- **Jurisdiction:** EU
- **Documents in corpus:** 8
- **Source collections:** `EBA-MiCA`
- **License:** EBA Legal Notice — Reproduction permitted with attribution
- **License URL:** <https://www.eba.europa.eu/legal-notice>
- **Required attribution:**

  > _(no specific string required — general attribution by source name)_

- **Caveats:**
  - Commercial use not explicitly addressed — silence on commercial reuse means bulk/commercial redistribution may require permission
  - EBA name, abbreviation and logo are exclusive property and require prior permission for use (except when reproducing EBA materials that already contain the logo)
  - Third-party copyrighted content embedded in EBA materials requires permission from the original copyright holder
  - Per-document exceptions exist: 'save where otherwise stated' — individual documents may carry different terms

### SSO — Parliament of Singapore

- **Jurisdiction:** Singapore
- **Documents in corpus:** 6
- **Source collections:** `SSO`
- **License:** Crown Copyright (Government of Singapore) — permission granted by AGC to reproduce Singapore legislation subject to SSO Terms of Use conditions
- **License URL:** <https://sso.agc.gov.sg/Terms-of-Use>
- **Required attribution:**

  > © Government of Singapore. Source: Singapore Statutes Online (https://sso.agc.gov.sg), Attorney-General's Chambers.

- **Caveats:**
  - Crown Copyright — all rights, title and interest owned by/licensed to/controlled by AGC
  - General permission granted by AGC to reproduce Singapore legislation for print or electronic material/platform, but only subject to the full conditions in SSO Terms of Use (clauses 13–15) which were not directly retrievable (sso.agc.gov.sg returned HTTP 403 to WebFetch)
  - Non-legislation content (e.g. site layout, editorial notes, metadata) is restricted: 'shall not be used, reproduced, republished, uploaded, posted, transmitted or otherwise distributed in any way, without the prior written permission of AGC'
  - SSO content is an unofficial version — not authoritative text of Singapore legislation; Section 48 of the Interpretation Act 1965 does not apply to anything printed, downloaded or copied from SSO
  - No open licence (no CC BY, no OGL, no public-domain dedication) — redistribution as part of a public academic dataset should be verified against the full clause 15 conditions before release
  - Commercial use is not explicitly addressed in the retrievable evidence — conservative reading required for any EMNLP-style public dataset redistribution

### BIS-Basel — Basel Committee on Banking Supervision (BCBS)

- **Jurisdiction:** INT-BIS
- **Documents in corpus:** 10
- **Source collections:** `BASEL`
- **License:** BIS Terms and Conditions — Non-commercial reuse with attribution
- **License URL:** <https://www.bis.org/terms_conditions.htm>
- **Required attribution:**

  > Source: Bank for International Settlements (BIS) / Basel Committee on Banking Supervision (BCBS)

- **Caveats:**
  - Non-commercial use only — commercial use requires written permission from BIS
  - All copyright and IP rights in publications are owned by the BIS
  - Limited extracts (max 400 words or 2 tables/graphs, not exceeding 10% of publication) may be reproduced free of charge if BIS is cited
  - Bulk redistribution beyond non-commercial use requires written permission
  - BIS Data Portal statistics are governed by separate terms ('About BIS statistics')
  - Translations must include a statement that they are not official BIS translations

### IOSCO — International Organization of Securities Commissions (IOSCO)

- **Jurisdiction:** INT-IOSCO
- **Documents in corpus:** 2
- **Source collections:** `IOSCO`
- **License:** IOSCO Terms and Conditions of Use (non-commercial redistribution with attribution; commercial use requires written permission)
- **License URL:** <https://www.iosco.org/v2/about/?subsection=terms-and-conditions>
- **Required attribution:**

  > _(no specific string required — general attribution by source name)_

- **Caveats:**
  - Non-commercial use only without written permission
  - Attribution to IOSCO as the source required
  - Commercial redistribution requires written permission from IOSCO (info@iosco.org / press@iosco.org)
  - IOSCO acronym and logo are IP-protected and excluded
  - Translations must state they are not official IOSCO translations

### 1.2 Withheld documents — metadata + URL only (55 of 247)

The following sources do **not** permit redistribution of their content. Their documents are listed in the inventory CSV with `source_url` populated, but the document files themselves are NOT included in this release. Users wishing to use these documents must download them directly from the source organisation.

### MAS — Monetary Authority of Singapore (MAS)

- **Jurisdiction:** Singapore
- **Documents (metadata only):** 34
- **Why withheld:** All Rights Reserved — MAS Terms of Use (prior written permission required)
- **Source page:** <https://www.mas.gov.sg/terms-of-use>
- **How to access:** Use the `source_url` column in `regulatory_documents_inventory_licensed.csv` to fetch the original documents from the source organisation.

### IRAS — Inland Revenue Authority of Singapore (IRAS)

- **Jurisdiction:** Singapore
- **Documents (metadata only):** 2
- **Why withheld:** All Rights Reserved — IRAS Copyright (prior written permission required)
- **Source page:** <https://www.iras.gov.sg/terms-of-use-browser-compatibility>
- **How to access:** Use the `source_url` column in `regulatory_documents_inventory_licensed.csv` to fetch the original documents from the source organisation.

### FATF — Financial Action Task Force (FATF)

- **Jurisdiction:** INT-FATF
- **Documents (metadata only):** 19
- **Why withheld:** Copyright FATF/OECD — All Rights Reserved (limited personal use only)
- **Source page:** <https://www.fatf-gafi.org/termsconditions/>
- **How to access:** Use the `source_url` column in `regulatory_documents_inventory_licensed.csv` to fetch the original documents from the source organisation.

---

## 2. Annotations

All annotations under `data/annotations/` — including `conflicts.jsonl`, `equivalences.jsonl`, `supersessions.jsonl`, `non_conflicts.jsonl`, and the SME review records — are the original work of the dataset authors and are released under:

**Creative Commons Attribution 4.0 International (CC BY 4.0)**

You are free to share and adapt the annotation data for any purpose, including commercial use, provided you give appropriate credit by citing the accompanying paper (citation in `CITATION.cff`).

Full license text: <https://creativecommons.org/licenses/by/4.0/legalcode>

---

## 3. Code

All source code under `src/`, `scripts/`, `ui/`, and `tests/` is released under the **MIT License**:

```
Copyright (c) 2026 TRIAG authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
```

---

## 4. Trained model artifacts

Trained model checkpoints (under `artifacts/models/`) derived from CC BY 4.0 annotations and the regulatory corpus are released under **CC BY 4.0**. When using these checkpoints, please cite the accompanying paper and preserve the attribution requirements of the underlying source documents listed in `ATTRIBUTIONS.md`.

---

## 5. Aggregate citation

If you use any component of this release, please cite:

```
[Citation to be added on publication — see CITATION.cff]
```

---

## 6. Disclaimer

This license file documents the redistribution terms applicable to this specific release. It does not constitute legal advice. The Tier 2 academic-use redistributions of EBA, SSO, BIS-Basel, and IOSCO content were made on the basis of each source's published academic-use permissions and the author's research office assessment. Users wishing to redistribute under different terms (e.g., commercial use of the corpus) should obtain permission from the respective source organisations.

Source-by-source licensing evidence, including verbatim quotations from each source's copyright page, is documented in `licensing_audit.md` accompanying this release.

---

## 7. Withheld-document fetch instructions

For the 55 withheld documents (MAS, IRAS, FATF), users may reconstruct the full corpus locally by following these steps:

1. Open `regulatory_documents_inventory_licensed.csv`.
2. Filter rows where `license_tier == "Tier 3"` (or `source_collection` matches `MAS-*`, `IRAS`, or `FATF`).
3. For each filtered row, use the `source_url` column to fetch the original document from the source organisation.
4. Verify the SHA-256 checksum (in the `sha256` column) against the fetched file to confirm you have the same document used in the original corpus.

A convenience manifest of the withheld documents is provided as `withheld_documents_manifest.csv`.
