
# SQL

- use `::` to cast, e.g. `::numeric`
- use `""` to distingush the name of the column with the built in function
  - e.g. `"first"`
- Sequence

    ```SQL
    SELECT -- either in group by or aggregated
    FROM
    WHERE
    INNER JOIN
    ON
    GROUP BY
    HAVING  -- after the data is grouped
    ORDER BY
- `count(*)`

- different sequence

    ```SQL
    -- writing
    SELECT 
    FROM
    JOIN
    WHERE
    GROUP BY
    HAVING 
    ORDER BY
    LIMIT

    -- EXECUTION
    FROM 
    JOIN
    WHERE
    GROUP BY
    HAVING
    SELECT -- *
    ORDER BY 
    LIMIT

- `CASE WHEN`

    ```SQL
    SELECT 
        CASE WHEN bedrooms > 3 THEN 'large'
            WHEN bedrooms = 3 THEN 'medium'
            ELSE 'small'
            END AS house_size,
        COUNT(*) as num_homes
    FROM housing_details
    GROUP BY 1 -- first column
    LIMIT 10

- `IN`

    ```SQL
    SELECT
        CASE WHEN postalcode IN ('44000', '01307')

- `JOIN`

    ```SQL
    INNER JOIN .. ON
    LEFT/RIGHT JOIN .. ON
    FULL OUTER JOIN .. ON

    SELECT .. UNION/INTERSECT/EXCEPT ..

- `ROUND`

  ```SQL
  ROUND(AVG(grade::numeric), 2)

- Nested queries

    ```SQL
    -- Median grade
    SELECT percentile_cont(0.5)
    WITHIN GROUP (ORDER BY grade)
    FROM housing_details;

    -- WHERE > single value
    WHERE grade > (
        SELECT percentile_cont(0.5)
        WITHIN GROUP (ORDER BY grade)
        FROM housing_details)

    -- WHERE IN Multiple values
    WHERE grade IN (SELECT ..)

    -- create a variable CTE
    WITH revenue_by_order AS (
        SELECT 
            order_id,
            SUM() AS revenue
        FROM order_details
        GROUP BY order_id
    )

- `RANK` ties - same rank; `DENSE_RANK` ties - consecutive

    ```SQL
    select count, 
        rank() over (order by count desc)


    -- rolling average
    with home_by_month AS (
        select year, month, count(*) as num_homes
        from main_home
        group by 1,2 
    )

    SELECT 
        year,
        month, 
        num_homes,
        SUM(num_homes) OVER (ORDER BY year, month) as  rolling_total -- rolling window,
        LAG(num_homes, 1) OVER (ORDER BY year, month) as lag1 -- prior record, help with month over month variance
        LEAD(num_homes, 1) OVER (ORDER BY year, month) as lead1 -- the next record,
        SUM(num_homes) OVER(PARTITION BY year order by year, month) as rolling_t_part -- within each year
    FROM home_by_month

- `VIEW`

    ```SQL
    CREATE OR REPLACE VIEW your_view_name AS
    -- your-queries

- `INDEX`

    ```SQL
    CREATE INDEX shipper_order_index ON
    orders (shipvia);


