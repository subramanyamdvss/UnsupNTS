set -e

#defining global hyperparameters 
maxitersteps=200000
loginterval=100
saveinterval=2000
cuda="--cuda"
unsup="--unsup"

#defining paths
ts=`pwd`
tsdata=$ts/tsdata
codepath=$ts/undreamt
model=$ts/modeldir

#creating new directories
mkdir -p "$model"
mkdir -p "$tsdata"


echo "${ts}"
echo "${tsdata}"



#UNTS - Using both classifier and discriminator 
batchsize=36
lr=0.00012
hidden=600
dropout=0.2
loginterval=100
saveinterval=200
embeddcom="$tsdata/fk.lower.vec"
pref="wgan.unsup.noadvcompl.control1.allclass.denoi.singleclassf.rho1.0.10k"
MONO=( tsdata/fkdifficpart-2m tsdata/fkeasypart-2m )
PARALLLEL=( tsdata/wiki-split.en.lower tsdata/wiki-split.sen.lower )



python3 "$codepath/train.py" --src_embeddings "$embeddcom" --trg_embeddings "$embeddcom"  --save "$model/model.$pref" \
 --batch $batchsize $cuda --disable_backtranslation --unsup --enable_mgan --add_control --easyprefix "tsdata/fkeasypart-2m" \
  --difficprefix "tsdata/fkdifficpart-2m" --start_save 9000 --stop_save 13000 



exit



#UNTS-10k - Using both classifier and discriminator with 10k parallel pairs 
batchsize=36
lr=0.00012
hidden=600
dropout=0.2
loginterval=100
saveinterval=200
embeddcom="$tsdata/fk.lower.vec"
pref="wgan.semisup10k-sel-6-4.noadvcompl.control1.allclass.denoi.singleclassf.rho1.0.10k"
MONO=( tsdata/fkdifficpart-2m tsdata/fkeasypart-2m )
PARALLLEL=( tsdata/wiki-split.en.lower tsdata/wiki-split.sen.lower )


python3 "$codepath/train.py" --src_embeddings "$embeddcom" --trg_embeddings "$embeddcom"  --save "$model/model.$pref"  $cuda\
 --src2trg "${PARALLLEL[0]}" "${PARALLLEL[1]}" --trg2src "${PARALLLEL[1]}" "${PARALLLEL[0]}" --disable_backtranslation \
 --enable_mgan --add_control --easyprefix "tsdata/fkeasypart-2m" --difficprefix "tsdata/fkdifficpart-2m" --start_save 6000 --stop_save 13000 



exit



# UNTS-10k - only with discriminator loss with 10k parallel pairs 
batchsize=36
lr=0.00012
hidden=600
dropout=0.2
loginterval=100
saveinterval=200
embeddcom="$tsdata/fk.lower.vec"
pref="wgan.semisup10k-sel-6-4.noadvcompl.control1.noclassf.denoi.singleclassf.rho1.0.10k"
MONO=( tsdata/fkdifficpart-2m tsdata/fkeasypart-2m )
PARALLLEL=( tsdata/wiki-split.en.lower tsdata/wiki-split.sen.lower )


python3 "$codepath/train.py" --src_embeddings "$embeddcom" --trg_embeddings "$embeddcom" --save "$model/model.$pref" --batch $batchsize   $cuda \
 --src2trg "${PARALLLEL[0]}" "${PARALLLEL[1]}" --trg2src "${PARALLLEL[1]}" "${PARALLLEL[0]}"  --disable_backtranslation --enable_mgan  --add_control \
--easyprefix "tsdata/fkeasypart-2m" --difficprefix "tsdata/fkdifficpart-2m" --noclassf --start_save 8000 --stop_save 13000 


exit

# UNMT using backtranslation and denoising - Artetxe et al 2018.
batchsize=32
lr=0.00012
hidden=600
dropout=0.2
loginterval=100
saveinterval=200
embeddcom="tsdata/fk.lower.vec"
pref="wgan.onlyback.denoi.back1.singleclassf.rho1.0.10k"
MONO=( tsdata/fkdifficpart-2m-1.lower tsdata/fkeasypart-2m-1.lower )

python3 "$codepath/train.py" --src "${MONO[0]}" --trg "${MONO[1]}" --src_embeddings "$embeddcom" --trg_embeddings "$embeddcom"  --save "$model/model.$pref" \
--batch $batchsize $cuda --unsup --start_save 18000 --stop_save 24000

exit


# UNTS-10k - only with classifier loss with 10k parallel pairs 
batchsize=36
lr=0.00012
hidden=600
dropout=0.2
loginterval=100
saveinterval=200
embeddcom="$tsdata/fk.lower.vec"
pref="wgan.semisup10k-sel-6-4.noadvcompl.control1.nodisc.denoi.singleclassf.rho1.0.10k"
MONO=( tsdata/fkdifficpart-2m tsdata/fkeasypart-2m )
PARALLLEL=( tsdata/wiki-split.en.lower tsdata/wiki-split.sen.lower )

python3 "$codepath/train.py" --src_embeddings "$embeddcom" --trg_embeddings "$embeddcom"  --save "$model/model.$pref" \
--batch $batchsize  $cuda --src2trg "${PARALLLEL[0]}" "${PARALLLEL[1]}" --trg2src "${PARALLLEL[1]}" "${PARALLLEL[0]}"  \
  --disable_backtranslation --enable_mgan  --add_control --easyprefix "tsdata/fkeasypart-2m" --difficprefix "tsdata/fkdifficpart-2m" --nodisc --start_save 6000 --stop_save 13000 

exit

