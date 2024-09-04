# Description: This script empties all the local run data, this
# is particularly useful when using volume mapping and  want to keep
# disk impact small while debugging or testing the code.

# Note this does not clear the references to this data in mongoDB
# so you may need to manually clear the database if you want to start
# fresh.

find ./colmap/data/inputs -type f -delete
find ./colmap/data/outputs -mindepth 1 -delete
find ./TensoRF/data/sfm_data -mindepth 1 -delete
find ./TensoRF/data/nerf_data -mindepth 1 -delete
find ./TensoRF/log -mindepth 1 -delete
find ./web-server/data/nerf -mindepth 1 -delete
find ./web-server/data/raw/videos -mindepth 1 -delete
find ./web-server/data/sfm -mindepth 1 -delete
find ./gaussian_splatting_reduced/data/nerf -mindepth 1 -delete
find ./gaussian_splatting_reduced/data/sfm -mindepth 1 -delete