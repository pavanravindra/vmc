for ((i=0; i<32; i++));
do

	dirName="trial${i}"

	if [ -d ${dirName} ]; then
		continue
	fi

	mkdir ${dirName}
	cd ${dirName}

	cp -r ../template/* .
	sbatch submit_folx.sh

	cd ..
	
done
