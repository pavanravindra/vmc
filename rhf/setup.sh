for N in 14;
do
	for rs in 1 2 5 10 20 50 100 110 200;
	do
		for kpts in 171;
		do

			filename="N${N}_rs${rs}"
		
			if [ -d ${filename} ]; then
				continue
			fi

			mkdir ${filename}

			cp run_rhf.py ${filename}
			cp submit.sh ${filename}

			cd ${filename}

			sed -i "s/xxxNxxx/${N}/g" run_rhf.py
			sed -i "s/xxxRSxxx/${rs}/g" run_rhf.py
			sed -i "s/xxxKxxx/${kpts}/g" run_rhf.py
			sbatch submit.sh

			cd ..

		done
	done
done
