# Week 2 Summary - Data Exploration

## Overview

Week 2 focused on validating and exploring the CRMLS sold-property dataset for California residential real estate price prediction. The analysis used the full available period from January 2025 through May 2026 and remained limited to descriptive EDA, data validation, and feasibility assessment.

The target analysis population was restricted to:

- `PropertyType = Residential`
- `PropertySubType = SingleFamilyResidence`

## What Was Done

- Loaded 17 monthly CRMLS sold files covering `2025-01` through `2026-05`.
- Combined monthly files into one dataset with a `source_month` field.
- Applied the required Residential + SingleFamilyResidence filter.
- Validated required columns including `ClosePrice`, `LivingArea`, `BedroomsTotal`, `BathroomsTotalInteger`, `LotSizeSquareFeet`, `PostalCode`, `CountyOrParish`, `CloseDate`, and `ListingContractDate`.
- Explored distributions for `ClosePrice`, log-transformed `ClosePrice`, `LivingArea`, bedrooms, bathrooms, and lot size.
- Checked missing values, non-positive values, implausible extremes, duplicate listing identifiers, and monthly/geographic coverage.

## Key Findings

- Raw data loaded: 352,354 rows.
- Filtered Residential + SingleFamilyResidence rows: 175,188 rows.
- The filtered dataset covers all 17 required months from January 2025 through May 2026.
- `ClosePrice` is highly right-skewed:
  - Median: about $895K
  - 75th percentile: about $1.43M
  - 95th percentile: about $3.2M
  - 99.5th percentile: about $8.8M
- Typical home size is concentrated in a reasonable single-family range:
  - Median `LivingArea`: about 1,814 sqft
  - Middle 50%: about 1,384 to 2,435 sqft
- Bedrooms and bathrooms are concentrated around standard single-family layouts:
  - Median bedrooms: 3
  - Median bathrooms: 2
- Geographic coverage is strongest in high-volume counties such as Los Angeles, Riverside, San Diego, San Bernardino, and Orange.

## Data Quality Issues

- `LotSizeSquareFeet` has the highest missingness among core EDA fields:
  - 3,056 missing rows
  - 94 non-positive values
  - extreme maximum above 2.0B sqft
- `LivingArea` has limited missingness but includes invalid/extreme values:
  - 92 missing rows
  - 78 non-positive values
  - maximum of 56,500 sqft
- `ClosePrice` has no missing values after filtering, but includes:
  - 1 non-positive value
  - extreme maximum near $989.5M
- Bedroom and bathroom fields are mostly complete but contain implausible extremes:
  - 108 non-positive bedroom counts
  - 80 non-positive bathroom counts
  - maximum values of 45 for both bedrooms and bathrooms
- Duplicate listing identifiers were detected:
  - 235 duplicate rows by `ListingKey` / `ListingId`
  - 186 duplicate rows by address + ZIP + close date

## Risks / Limitations

- The dataset is suitable for Week 2 EDA, but several fields require cleaning decisions before later stages.
- Extreme `ClosePrice`, `LivingArea`, bedroom, bathroom, and lot-size values may represent data entry errors, special property types, or valid but rare luxury properties.
- Lot size is especially risky because the distribution is highly skewed and geography-dependent.
- Geographic volume is uneven, so low-volume ZIP codes may be less reliable for market-level conclusions.
- Week 2 does not remove, impute, or transform records; it only flags issues for follow-up.

## Next Steps (Week 3)

- Define documented cleaning rules for missing, invalid, duplicated, and extreme records.
- Decide which rows are removed, corrected, imputed, or flagged, with justification.
- Validate date fields and lifecycle consistency before downstream analysis.
- Produce a cleaned dataset for the next project phase.
- Keep all cleaning decisions reproducible and documented.
