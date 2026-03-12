-- =============================================================================
-- analytics/sql_queries.sql
-- =============================================================================
-- Healthcare Data Analytics — SQL Query Library
--
-- Purpose
-- -------
-- A curated collection of analytical SQL queries for the AI Healthcare Data
-- Platform.  The queries are written in ANSI SQL and have been tested against
-- Apache Spark SQL / Delta Lake and PostgreSQL 14+.
--
-- Table assumptions
-- -----------------
--   clinical_records   : Cleansed clinical encounter data
--                        (patient_id, encounter_date, diagnosis_code,
--                         encounter_id, provider_id, facility_id,
--                         admission_type, discharge_disposition,
--                         length_of_stay_days, total_charge)
--
--   genomic_variants   : Filtered genomic variant records
--                        (sample_id, chromosome, position, ref, alt,
--                         qual, depth, consequence, ingested_at)
--
--   patients           : Patient demographics
--                        (patient_id, age_at_registration, gender,
--                         ethnicity, zip_code, insurance_type)
--
--   risk_scores        : ML model output
--                        (patient_id, scored_at, risk_score,
--                         risk_category, model_version)
--
-- =============================================================================

-- =============================================================================
-- SECTION 1 — Clinical Data Overview
-- =============================================================================

-- 1.1  Total encounters per month
-- Trend of hospital activity over time.
SELECT
    DATE_TRUNC('month', encounter_date)          AS month,
    COUNT(*)                                      AS total_encounters,
    COUNT(DISTINCT patient_id)                    AS unique_patients,
    ROUND(AVG(length_of_stay_days), 2)            AS avg_los_days
FROM clinical_records
GROUP BY DATE_TRUNC('month', encounter_date)
ORDER BY month;


-- 1.2  Top 20 most frequent diagnoses (ICD-10)
-- Identifies disease burden across the patient population.
SELECT
    diagnosis_code,
    COUNT(*)                                      AS encounter_count,
    COUNT(DISTINCT patient_id)                    AS unique_patients,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2) AS pct_of_total
FROM clinical_records
GROUP BY diagnosis_code
ORDER BY encounter_count DESC
LIMIT 20;


-- 1.3  Readmission rate within 30 days
-- Flags patients re-admitted within 30 days of a prior discharge.
WITH ordered_encounters AS (
    SELECT
        patient_id,
        encounter_id,
        encounter_date,
        LAG(encounter_date) OVER (
            PARTITION BY patient_id ORDER BY encounter_date
        )                                         AS prev_encounter_date
    FROM clinical_records
),
readmissions AS (
    SELECT
        patient_id,
        encounter_id,
        encounter_date,
        prev_encounter_date,
        DATEDIFF(encounter_date, prev_encounter_date) AS days_since_last
    FROM ordered_encounters
    WHERE prev_encounter_date IS NOT NULL
)
SELECT
    COUNT(*)                                      AS total_encounters,
    SUM(CASE WHEN days_since_last <= 30 THEN 1 ELSE 0 END)
                                                  AS readmissions_30d,
    ROUND(
        100.0 * SUM(CASE WHEN days_since_last <= 30 THEN 1 ELSE 0 END) / COUNT(*),
        2
    )                                             AS readmission_rate_pct
FROM readmissions;


-- 1.4  Average length of stay by admission type and discharge disposition
SELECT
    admission_type,
    discharge_disposition,
    COUNT(*)                                      AS encounters,
    ROUND(AVG(length_of_stay_days), 2)            AS avg_los_days,
    ROUND(MIN(length_of_stay_days), 2)            AS min_los_days,
    ROUND(MAX(length_of_stay_days), 2)            AS max_los_days
FROM clinical_records
GROUP BY admission_type, discharge_disposition
ORDER BY avg_los_days DESC;


-- =============================================================================
-- SECTION 2 — Patient Demographics & Health Equity
-- =============================================================================

-- 2.1  Encounter volume by age group and gender
SELECT
    CASE
        WHEN p.age_at_registration < 18  THEN 'Paediatric (<18)'
        WHEN p.age_at_registration < 40  THEN 'Young Adult (18–39)'
        WHEN p.age_at_registration < 65  THEN 'Middle-Aged (40–64)'
        ELSE                                  'Senior (65+)'
    END                                          AS age_group,
    p.gender,
    COUNT(c.encounter_id)                         AS total_encounters,
    COUNT(DISTINCT c.patient_id)                  AS unique_patients
FROM clinical_records  c
JOIN patients           p ON c.patient_id = p.patient_id
GROUP BY age_group, p.gender
ORDER BY age_group, p.gender;


-- 2.2  Insurance type distribution and average charge
SELECT
    p.insurance_type,
    COUNT(DISTINCT p.patient_id)                  AS patient_count,
    COUNT(c.encounter_id)                         AS encounters,
    ROUND(AVG(c.total_charge), 2)                 AS avg_charge,
    ROUND(SUM(c.total_charge), 2)                 AS total_revenue
FROM clinical_records c
JOIN patients          p ON c.patient_id = p.patient_id
GROUP BY p.insurance_type
ORDER BY patient_count DESC;


-- =============================================================================
-- SECTION 3 — Genomic Variant Analysis
-- =============================================================================

-- 3.1  Variant consequence distribution
SELECT
    COALESCE(consequence, 'unannotated')          AS consequence_type,
    COUNT(*)                                      AS variant_count,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2) AS pct_of_total
FROM genomic_variants
GROUP BY consequence_type
ORDER BY variant_count DESC;


-- 3.2  High-impact variants per sample (quality ≥ 30, depth ≥ 10)
-- Useful for identifying samples with a high burden of damaging mutations.
SELECT
    sample_id,
    COUNT(*)                                      AS high_impact_variants,
    COUNT(DISTINCT chromosome)                    AS chromosomes_affected
FROM genomic_variants
WHERE
    qual    >= 30
    AND depth  >= 10
    AND consequence IN (
        'stop_gained', 'frameshift_variant', 'splice_acceptor_variant',
        'splice_donor_variant', 'missense_variant'
    )
GROUP BY sample_id
ORDER BY high_impact_variants DESC;


-- 3.3  Variant density per chromosome (normalised to chromosome length)
-- Chromosome lengths (GRCh38) in Mbp — adjust as required.
WITH chrom_lengths (chromosome, length_mbp) AS (
    VALUES
        ('1',249), ('2',242), ('3',198), ('4',190), ('5',181),
        ('6',170), ('7',159), ('8',145), ('9',138), ('10',133),
        ('11',135), ('12',133), ('13',114), ('14',107), ('15',101),
        ('16',90),  ('17',83),  ('18',80),  ('19',58),  ('20',64),
        ('21',46),  ('22',50),  ('X',156),  ('Y',57)
),
variant_counts AS (
    SELECT chromosome, COUNT(*) AS variant_count
    FROM   genomic_variants
    GROUP  BY chromosome
)
SELECT
    vc.chromosome,
    vc.variant_count,
    cl.length_mbp,
    ROUND(CAST(vc.variant_count AS DECIMAL) / cl.length_mbp, 4)
                                                  AS variants_per_mbp
FROM variant_counts vc
JOIN chrom_lengths   cl ON vc.chromosome = cl.chromosome
ORDER BY variants_per_mbp DESC;


-- =============================================================================
-- SECTION 4 — Risk Scoring & Model Performance
-- =============================================================================

-- 4.1  Risk category distribution (latest score per patient)
WITH latest_scores AS (
    SELECT DISTINCT ON (patient_id)
        patient_id,
        risk_score,
        risk_category,
        scored_at,
        model_version
    FROM risk_scores
    ORDER BY patient_id, scored_at DESC
)
SELECT
    risk_category,
    model_version,
    COUNT(*)                                      AS patient_count,
    ROUND(AVG(risk_score), 4)                     AS avg_risk_score,
    ROUND(MIN(risk_score), 4)                     AS min_risk_score,
    ROUND(MAX(risk_score), 4)                     AS max_risk_score
FROM latest_scores
GROUP BY risk_category, model_version
ORDER BY avg_risk_score DESC;


-- 4.2  High-risk patients with recent encounters
-- Operationally useful to flag patients that need follow-up.
WITH latest_scores AS (
    SELECT DISTINCT ON (patient_id)
        patient_id,
        risk_score,
        risk_category,
        scored_at
    FROM risk_scores
    ORDER BY patient_id, scored_at DESC
),
recent_encounters AS (
    SELECT patient_id, MAX(encounter_date) AS last_encounter_date
    FROM   clinical_records
    GROUP  BY patient_id
)
SELECT
    ls.patient_id,
    p.age_at_registration,
    p.gender,
    p.insurance_type,
    ls.risk_score,
    ls.risk_category,
    re.last_encounter_date
FROM latest_scores     ls
JOIN patients          p  ON ls.patient_id = p.patient_id
JOIN recent_encounters re ON ls.patient_id = re.patient_id
WHERE
    ls.risk_category = 'HIGH'
    AND re.last_encounter_date >= CURRENT_DATE - INTERVAL '90 days'
ORDER BY ls.risk_score DESC;


-- 4.3  Risk score trend over time (monthly average per category)
SELECT
    DATE_TRUNC('month', scored_at)               AS month,
    risk_category,
    model_version,
    COUNT(DISTINCT patient_id)                    AS scored_patients,
    ROUND(AVG(risk_score), 4)                     AS avg_risk_score
FROM risk_scores
GROUP BY DATE_TRUNC('month', scored_at), risk_category, model_version
ORDER BY month, risk_category;


-- =============================================================================
-- SECTION 5 — Integrated Clinical + Genomic Insights
-- =============================================================================

-- 5.1  Patients with high-impact variants AND high clinical risk scores
SELECT
    p.patient_id,
    p.age_at_registration,
    p.gender,
    hiv.high_impact_count,
    rs.risk_score,
    rs.risk_category
FROM patients p
JOIN (
    SELECT sample_id AS patient_id, COUNT(*) AS high_impact_count
    FROM   genomic_variants
    WHERE  consequence IN (
               'stop_gained', 'frameshift_variant', 'missense_variant'
           )
    AND    qual >= 30
    GROUP  BY sample_id
) hiv ON p.patient_id = hiv.patient_id
JOIN (
    SELECT DISTINCT ON (patient_id) patient_id, risk_score, risk_category
    FROM   risk_scores
    ORDER  BY patient_id, scored_at DESC
) rs ON p.patient_id = rs.patient_id
WHERE rs.risk_category = 'HIGH'
ORDER BY hiv.high_impact_count DESC, rs.risk_score DESC;
