sourcedir=/home/mperry/projects/Moore_food/aez_climate_prediction_try2/2050s
outdir=/home/mperry/projects/Moore_food/aez_climate_prediction_try2/2050s_2

sourcefiles=(p_ph_c \
tmin12c \
irr_lands \
gt_demc \
tmax8c \
pmean_wntrc \
grwsnc \
d2u2c \
hy200 \
pmean_sumrc)

for item in ${sourcefiles[*]}
do
    echo
    echo $sourcedir/$item
    gdalwarp -of HFA -tr 1000 1000 -te -2017872 -1086844 -856133 709691 -r bilinear $sourcedir/$item/hdr.adf $outdir/${item}.img
done

echo "zones"
zones=iso_maj3-27
gdalwarp -of HFA -tr 1000 1000 -te -2017872 -1086844 -856133 709691 -r near $sourcedir/$zones/hdr.adf $outdir/${zones}.img
