-- @block
DROP TABLE IF EXISTS features
-- @block
CREATE TABLE features(
    id INT PRIMARY KEY AUTO_INCREMENT,
    center_x DOUBLE NOT NULL,
    center_y DOUBLE NOT NULL,
    center_z DOUBLE NOT NULL,
    normal_x DOUBLE NOT NULL,
    normal_y DOUBLE NOT NULL,
    normal_z DOUBLE NOT NULL,
    area DOUBLE NOT NULL,
    num_neighbors INT NOT NULL,
    point_1_x DOUBLE NOT NULL,
    point_1_y DOUBLE NOT NULL,
    point_1_z DOUBLE NOT NULL,
    point_2_x DOUBLE NOT NULL,
    point_2_y DOUBLE NOT NULL,
    point_2_z DOUBLE NOT NULL,
    point_3_x DOUBLE NOT NULL,
    point_3_y DOUBLE NOT NULL,
    point_3_z DOUBLE NOT NULL
)
-- @block
INSERT INTO features(center_x, center_y, center_z, normal_x, normal_y, normal_z, area, num_neighbors, point_1_x, point_1_y, point_1_z, point_2_x, point_2_y, point_2_z, point_3_x, point_3_y, point_3_z)
VALUES 
    (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 2.1, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0),
    (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 2.1, 1.0, 2.0, 3.0, 4.0, 5.5, 6.0, 7.0, 8.0, 9.0),
    (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.1, 2.1, 1.0, 2.0, 3.0, 5.0, 5.0, 6.0, 7.0, 8.0, 9.0),
    (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 2.1, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 5.0, 8.0, 9.0),
    (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 2.1, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 5.0, 9.0)
-- @block
SELECT id, center_x, center_y, center_z FROM features
WHERE area = 7.0 AND id < 10
ORDER BY id DESC
LIMIT 5