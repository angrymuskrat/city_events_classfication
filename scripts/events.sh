PGPASSWORD=secretpwd psql -h 10.9.14.132 -U secretuser -d $1 -c "\copy (select id, title, tags, postcodes from events) to '$1_events.csv' csv header;"
