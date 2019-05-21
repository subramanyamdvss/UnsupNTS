set -e

#defining paths
ts=( `pwd` )
src=en
tgt=sen
tsdata=$ts/tsdata
codepath=$ts/undreamt
ntsevalcode=$ts/utils/evaluate.py
gendir=$ts/predictions
genfile=$gendir/gen_lower.$tgt
model=$ts/modeldir
logdir="$ts/logs/TS.GEN"
file="test"
srcfile=tsdata/$file.en

#creating new directories
mkdir -p "$gendir"
mkdir -p "$model"
mkdir -p "$logdir"



#Generating simplifications
nlines=( 10000 )
control_nums=( 1 )
for ncontrol in "${control_nums[@]}"
do
	for nline in "${nlines[@]}"
	do
	 	modelnum=$nline
    	pref="wgan.semisup10k-sel-6-4.noadvcompl.control1.allclass.denoi.singleclassf.rho1.0.10k"
		noise=0.0
		pref="$pref"
		modelfile=$model/model.$pref.it$modelnum.src2trg.pth
		echo model.$pref.it$modelnum.src2trg.pth
		python3 -u "$codepath/translate.py" "$modelfile" --input "$srcfile" --output "$genfile.src2trg.${pref}.$nline.$file"  --noise $noise \
		--batch_size 100 --ncontrol $ncontrol \
		>> "$logdir/out.src2trg.$pref" 
		
	done 
done






#Evaluating the Simplifications
nlines=( 10000 )
control_nums=( 1 )
for ncontrol in "${control_nums[@]}"
do
	for nline in "${nlines[@]}"
	do
    	pref="wgan.semisup10k-sel-6-4.noadvcompl.control1.allclass.denoi.singleclassf.rho1.0.10k"
		noise=0.0
	 	modelnum=$nline
		modelfile=$modeldir/model.$pref.it$modelnum.src2trg.pth
		echo model.$pref.it$modelnum.src2trg.pth
		genf=$genfile.src2trg.${pref}.$modelnum.$file
		python  predictions/noredund.py < "$genf" > "${genf}.noredund"
		genf="$genf.noredund"
		python utils/lev.py --input "$genf" --source "$srcfile" 
		python utils/fk_ts_ds.py -i "$genf" -src "$srcfile" 
		mkdir -p tmp
		cp "$genf" tmp/
		python2 "$ntsevalcode" "$srcfile" "$tsdata/references.tsv" tmp 
		rm -rf tmp

	done 
done

