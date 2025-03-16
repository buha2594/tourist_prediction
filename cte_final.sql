USE tourism;

WITH cte AS (
    SELECT
        B.userid,
        B.visityear,
        B.visitMonth,
        B.visitmodeID,
        B.attractionid,
        B.rating,
        A.Attraction,
        A.AttractionAddress,
        A.AttractionCityId,  -- This is the city where the attraction is located
        A.attractiontypeid
    FROM transaction B
    INNER JOIN item A ON A.attractionid = B.attractionid
),
cte2 AS (
    SELECT
        A.*,
        B.contenentid,
        B.regionid,
        B.countryid,
        B.cityid
    FROM cte A
    LEFT JOIN user B ON A.userid = B.userid
),
cte3 AS (
    SELECT A.*, B.attractiontype FROM cte2 A
    LEFT JOIN type B ON A.attractiontypeid = B.attractiontypeid
),
cte4 AS (
    SELECT A.*, B.visitmode FROM cte3 A
    LEFT JOIN mode B ON A.visitmodeID = B.VisitModeId
),
cte5 AS (
    SELECT A.*, B.cityname AS UserCity FROM cte4 A
    LEFT JOIN city B ON A.cityid = B.cityid
),
cte6 AS (
    SELECT A.*, B.country AS UserCountry FROM cte5 A
    LEFT JOIN country B ON A.countryid = B.countryid
),
cte7 AS (
    SELECT A.*, B.region AS UserRegion FROM cte6 A
    LEFT JOIN region B ON A.RegionId = B.RegionId
),
cte8 AS (
    SELECT A.*, B.contenent AS UserContinent FROM cte7 A
    LEFT JOIN continent B ON A.ContenentId = B.ContenentId
),
cte9 AS (
    -- Getting Attraction City, Country, and Region
    SELECT
        A.*,
        C.cityname AS AttractionCityName,  -- Get attraction's city name
        CO.country AS AttractionCountry,  -- Get attraction's country name
        R.region AS AttractionRegion      -- Get attraction's region name
    FROM cte8 A
    LEFT JOIN city C ON A.AttractionCityId = C.CityId
    LEFT JOIN country CO ON C.CountryId = CO.CountryId
    LEFT JOIN region R ON CO.RegionId = R.RegionId
)

-- Exporting Data to CSV
SELECT
    'userid', 'visityear', 'visitMonth', 'visitmodeID', 'attractionid', 'rating',
    'attraction', 'AttractionAddress', 'AttractionCityId', 'attractiontypeid',
    'contenentid', 'regionid', 'countryid', 'cityid', 'attractiontype', 'visitmode',
    'UserCity', 'UserCountry', 'UserRegion', 'UserContinent', -- User's details
    'AttractionCityName', 'AttractionCountry', 'AttractionRegion'  -- Attraction's details
UNION ALL
SELECT
    userid, visityear, visitMonth, visitmodeID, attractionid, rating,
    attraction, AttractionAddress, AttractionCityId, attractiontypeid,
    contenentid, regionid, countryid, cityid, attractiontype, visitmode,
    UserCity, UserCountry, UserRegion, UserContinent,
    AttractionCityName, AttractionCountry, AttractionRegion
FROM cte9
INTO OUTFILE 'C:/ProgramData/MySQL/MySQL Server 8.0/Uploads/debug6.csv'
FIELDS TERMINATED BY ','
ENCLOSED BY '"'
LINES TERMINATED BY '\n';
