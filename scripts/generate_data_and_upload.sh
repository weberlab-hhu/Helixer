#! /bin/bash

local_main_folder="/home/felix/Desktop/new_best_geenuff/"
remote_main_folder="/dev/shm/helixer_new/"
db_path='/home/felix/Desktop/full_geenuff.sqlite3'

for size in $@
do
	echo -e "\nsize: "$size"000"
	local_folder="$local_main_folder"h5_data_"$size"k

	rm -r "$local_folder"/*

	# ../export.py --db-path-in $db_path --out-dir $local_folder --chunk-size "$size"000 --genomes Creinhardtii,Gmax,Tcacao,Mpolymorpha;
	../export.py --db-path-in $db_path --out-dir $local_folder --chunk-size "$size"000 --genomes Creinhardtii

	rsync -vzr --progress $local_folder felix-stiehler@134.99.200.63:"$remote_main_folder"
done
