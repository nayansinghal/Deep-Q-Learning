for d in logs/*/; do
	echo "$d"
	grep 'Reward' $d/log1.txt | cut -d ' ' -f6 > $d/log1_cut.txt
done
