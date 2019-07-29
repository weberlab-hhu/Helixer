#! /bin/bash

db_path='/home/felix/Desktop/full_geenuff.sqlite3'
local_main_folder_input="/home/felix/git/GeenuFF/tmp/"
local_main_folder_output="/home/felix/Desktop/data/single_genomes/"
remote_main_folder="/dev/shm/helixer/data/single_genomes/"

len_in_k=10
coordinate_chance=1.0

for genome in $(ls -1 $local_main_folder_input)
do
	folder_out="$local_main_folder_output$genome/h5_data_$len_in_k""k"
	echo -e "\nGenerating $genome"
	../export.py --db-path-in $db_path --out-dir $folder_out --chunk-size "$len_in_k"000 --coordinate-chance $coordinate_chance --genomes $genome --only-test-set

	echo -e "\nSending $genome"
	rsync -rvz --progress "$folder_out" felix-stiehler@134.99.200.63:"$remote_main_folder$genome/"
done
