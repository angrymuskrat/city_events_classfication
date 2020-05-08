#!/bin/bash

sudo -u postgres psql -d $1 -c "\copy (select id, caption from posts where timestamp < 1483228800) to '$1_posts2016.csv' csv header;"
#sudo -u postgres psql -d $1 -c "\copy (select id, caption from posts where timestamp between 1483228800 and 1514764800) to '$1_posts2017.csv' csv header;"
#sudo -u postgres psql -d $1 -c "\copy (select id, caption from posts where timestamp between 1514764800 and 1546300800) to '$1_posts2018.csv' csv header;"
#sudo -u postgres psql -d $1 -c "\copy (select id, caption from posts where timestamp between 1546300800 and 1577836800) to '$1_posts2019.csv' csv header;"
#sudo -u postgres psql -d $1 -c "\copy (select id, caption from posts where timestamp > 1577836800) to '$1_posts2020.csv' csv header;"
