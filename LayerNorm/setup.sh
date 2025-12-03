while read -r rs lrMin0 lrMax0 muMin0 muMax0;
do

	for ((i=0; i<4; i++));
	do

		dirName="reference/rs${rs}_trial${i}"

		if [ -d ${dirName} ]; then
			continue
		fi

		mkdir ${dirName}
		cd ${dirName}

		cp -r ../../template/* .
		sed -i "s/xxxRSxxx/${rs}/g" train_and_eval.py
		sed -i "s/xxxLRMIN0xxx/${lrMin0}/g" train_and_eval.py
		sed -i "s/xxxLRMAX0xxx/${lrMax0}/g" train_and_eval.py

		sbatch submit_training.sh 

		cd ../..

	done

done < RS_MP_SETTINGS
