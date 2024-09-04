# Description: This script empties all the local run data, this
# is particularly useful when using volume mapping and  want to keep
# disk impact small while debugging or testing the code.

# Note this does not clear the references to this data in mongoDB
# so you may need to manually clear the database if you want to start
# fresh.

find ./sfm-worker/data/inputs -type f -delete
find ./sfm-worker/data/outputs -mindepth 1 -delete

find ./nerf-worker/data/nerf -mindepth 1 -delete
find ./nerf-worker/data/sfm -mindepth 1 -delete

find ./web-server/data/nerf -mindepth 1 -delete
find ./web-server/data/raw/videos -mindepth 1 -delete
find ./web-server/data/sfm -mindepth 1 -delete

find ./go-web-server/data/nerf -mindepth 1 -delete
find ./go-web-server/data/raw/videos -mindepth 1 -delete
find ./go-web-server/data/sfm -mindepth 1 -delete
