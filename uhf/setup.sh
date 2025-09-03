for N in 14; # 54;
do
	for rs in 1 2 5 10 20 50 100 110 200;
	do
		for kpts in 27; # 171;
		do

			filename="N${N}_rs${rs}_k${kpts}"
		
			if [ -d ${filename} ]; then
				continue
			fi

			mkdir ${filename}

			cp run_uhf.py ${filename}
			cp submit.sh ${filename}

			cd ${filename}

			sed -i "s/xxxNxxx/${N}/g" run_uhf.py
			sed -i "s/xxxRSxxx/${rs}/g" run_uhf.py
			sed -i "s/xxxKxxx/${kpts}/g" run_uhf.py
			sbatch submit.sh

			cd ..

		done
	done
done
