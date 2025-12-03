for ((i=1; i<=20; i++));
do

	parameterTimestep="${i}00"
	filename="eval_${parameterTimestep}.py"
	submitFile="submit_${parameterTimestep}.sh"

	cp eval_template.py ${filename}
	cp submit_template.sh ${submitFile}

	sed -i "s/xxxTODOxxx/${parameterTimestep}/g" ${filename}
	sed -i "s/xxxFILExxx/${filename}/g" ${submitFile}

	sbatch ${submitFile}

done
