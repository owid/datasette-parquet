CREATE TABLE fixtures (date date, ts timestamp);
INSERT INTO fixtures (date, ts) VALUES ('2023-01-01', '2023-01-02T03:04:05Z');
CREATE TABLE posts (slug varchar, published varchar);
INSERT INTO posts (slug, published) VALUES ('a', '2023-03-04'), ('b', 'not a date');
