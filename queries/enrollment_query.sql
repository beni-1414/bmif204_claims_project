
WITH first_fills AS (
    SELECT 
        r.MemberUID,
        r.ndc11code,
        MIN(r.filldate) AS first_fill_date
    FROM [InovalonSample1M].[dbo].[RxClaim] r
    WHERE r.supplydayscount IS NOT NULL
    GROUP BY r.MemberUID, r.ndc11code
),
ranked_drugs AS (
    SELECT 
        MemberUID,
        ndc11code,
        first_fill_date,
        ROW_NUMBER() OVER (
            PARTITION BY MemberUID ORDER BY first_fill_date
        ) AS drug_rank
    FROM first_fills
),
polypharmacy_index AS (
    SELECT 
        MemberUID,
        first_fill_date AS index_date
    FROM ranked_drugs
    WHERE drug_rank = 5
),
eligible_members AS (
    SELECT 
        p.MemberUID,
        p.index_date,
        m.birthyear,
        (YEAR(p.index_date) - m.birthyear) AS age_at_index
    FROM polypharmacy_index p
    JOIN [InovalonSample1M].[dbo].[Member] m
      ON p.MemberUID = m.MemberUID
    WHERE (YEAR(p.index_date) - m.birthyear) >= 65
),
rx_filtered AS (
    SELECT 
        r.MemberUID,
        r.filldate,
        r.ndc11code,
        r.supplydayscount
    FROM [InovalonSample1M].[dbo].[RxClaim] r
    JOIN eligible_members em
      ON r.MemberUID = em.MemberUID
    WHERE r.supplydayscount IS NOT NULL
      AND r.filldate BETWEEN DATEADD(DAY, -180, em.index_date) AND DATEADD(YEAR, 1, em.index_date)
),
adverse_events AS (
    SELECT 
        c.MemberUID,
        cc.ServiceDate AS event_date,
        cc.CodeType,
        cc.CodeValue
    FROM [InovalonSample1M].[dbo].[Claim] c
    JOIN [InovalonSample1M].[dbo].[ClaimCode] cc
       ON c.ClaimUID = cc.ClaimUID
    JOIN eligible_members em
       ON c.MemberUID = em.MemberUID
    WHERE cc.CodeType IN (17,18,22,23,24)  -- ICD10 only
      AND (cc.CodeValue LIKE 'T36%' OR cc.CodeValue LIKE 'T37%' OR cc.CodeValue LIKE 'T38%' OR cc.CodeValue LIKE 'T39%' OR cc.CodeValue LIKE 'T40%' OR cc.CodeValue LIKE 'T41%' OR cc.CodeValue LIKE 'T42%' OR cc.CodeValue LIKE 'T43%' OR cc.CodeValue LIKE 'T44%' OR cc.CodeValue LIKE 'T45%' OR cc.CodeValue LIKE 'T46%' OR cc.CodeValue LIKE 'T47%' OR cc.CodeValue LIKE 'T48%' OR cc.CodeValue LIKE 'T49%' OR cc.CodeValue LIKE 'T50%' OR cc.CodeValue LIKE 'R296%' OR cc.CodeValue LIKE 'R410%' OR cc.CodeValue LIKE 'F30%' OR cc.CodeValue LIKE 'F31%' OR cc.CodeValue LIKE 'F32%' OR cc.CodeValue LIKE 'F39%' OR cc.CodeValue LIKE 'R42%' OR cc.CodeValue LIKE 'R44%' OR cc.CodeValue LIKE 'R502%' OR cc.CodeValue LIKE 'T886%' OR cc.CodeValue LIKE 'T887%' OR cc.CodeValue LIKE 'N141%' OR cc.CodeValue LIKE 'D590%' OR cc.CodeValue LIKE 'D592%' OR cc.CodeValue LIKE 'D611%' OR cc.CodeValue LIKE 'J704%' OR cc.CodeValue LIKE 'K71%' OR cc.CodeValue LIKE 'K711%' OR cc.CodeValue LIKE 'K712%' OR cc.CodeValue LIKE 'K716%' OR cc.CodeValue LIKE 'K719%' OR cc.CodeValue LIKE 'L561%')
)

SELECT  e.MemberUID, e.effectivedate, e.terminationdate
FROM [InovalonSample1M].[dbo].[MemberEnrollment] e
JOIN eligible_members em ON e.MemberUID = em.MemberUID
