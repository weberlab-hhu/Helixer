#! /bin/bash

local_main_folder="/home/felix/Desktop/data/eight_genomes/"
remote_main_folder_clc="/home/felix-stiehler/Desktop/data/eight_genomes/"
remote_main_folder_cluster="/scratch_gs/festi100/data/eight_genomes/"
db_path='/home/felix/Desktop/full_geenuff.sqlite3'

for size in $@
do
	echo -e "\nsize: "$size"000"
	local_folder="$local_main_folder"h5_data_"$size"k

	rm -r "$local_folder"/*

	../export.py --db-path-in $db_path --out-dir $local_folder --chunk-size "$size"000 --genomes Athaliana,Bdistachyon,Creinhardtii,Gmax,Mguttatus,Mpolymorpha,Ptrichocarpa,Sitalica

	rsync -vzr --progress $local_folder felix-stiehler@134.99.200.63:"$remote_main_folder_clc"
	rsync -vzr --progress $local_folder festi100@hpc.rz.uni-duesseldorf.de:"$remote_main_folder_cluster"
done
