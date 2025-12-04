SELECT
    r.MemberUID
INTO <YOUR_USERNAME>.dbo.elegible_members
FROM RxClaim r
LEFT JOIN dbo.QueryDrugs o
    ON r.ndc11code = o.ndc11code
WHERE r.supplydayscount IS NOT NULL
GROUP BY r.MemberUID
HAVING
    MAX(CASE WHEN o.ndc11code IS NOT NULL THEN 1 ELSE 0 END) = 1
    AND COUNT(DISTINCT r.ndc11code) >= 3;