#! /bin/bash

db_path='/home/felix/Desktop/full_geenuff.sqlite3'
local_main_folder_input="/home/felix/git/GeenuFF/tmp/"
local_main_folder_output="/home/felix/Desktop/data/single_genomes/"
remote_main_folder="/dev/shm/helixer/data/single_genomes/"

chunk_size=5000
coordinate_chance=1.0

for genome in $(ls -1 $local_main_folder_input)
do
	echo -e "\nGenerating $genome"
	../export.py --db-path-in $db_path --out-dir "$local_main_folder_output$genome" --chunk-size $chunk_size --coordinate-chance $coordinate_chance --genomes $genome --only-test-set

	echo -e "\nSending $genome"
	rsync -vz --progress "$local_main_folder_output$genome/test_data.h5" felix-stiehler@134.99.200.63:"$remote_main_folder$genome/"
done

for genome in $(ls -1 $local_main_folder_input)
do
	echo ""
	./get_length_and_error_rate.py --data "$local_main_folder_output$genome/test_data.h5"
done
