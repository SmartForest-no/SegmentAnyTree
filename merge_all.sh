
#!/bin/bash

# FILEPATH: /home/nibio/mutable-outside-world/merge_all.sh

# Check if folder name is provided as argument
if [ -z "$1" ]
    then
        echo "Please provide the folder name as an argument"
        exit 1
fi

# Set folder name from argument
folder_name=$1

# kutter all filer i tmp_for_merge
rm tmp_for_merge/*

# kjore merge_inference_results.py for alle resultater
python3 nibio_inference/merge_inference_results.py -i ~/data/"$folder_name"/results_/ -o tmp_for_merge/merged_results_org.csv
# remove the top line
sed -i '1d' tmp_for_merge/merged_results_org.csv

# put the lines in csv in the following order (austrian_plot, English_plot, for_instance, german_plot, mls)
# sort them by the first column alphabetically
sort -t, -k1,1 tmp_for_merge/merged_results_org.csv > tmp_for_merge/merged_results_org_sorted.csv
mv tmp_for_merge/merged_results_org_sorted.csv tmp_for_merge/merged_results_org.csv


# kjore merge_inference_results.py for alle resultater
python3 nibio_inference/merge_inference_results.py -i ~/data/"$folder_name"/results_1000/ -o tmp_for_merge/merged_results_1000.csv
# remove the top line
sed -i '1d' tmp_for_merge/merged_results_1000.csv

# sort them by the first column alphabetically
sort -t, -k1,1 tmp_for_merge/merged_results_1000.csv > tmp_for_merge/merged_results_1000_sorted.csv
mv tmp_for_merge/merged_results_1000_sorted.csv tmp_for_merge/merged_results_1000.csv

# kjore merge_inference_results.py for alle resultater
python3 nibio_inference/merge_inference_results.py -i ~/data/"$folder_name"/results_500/ -o tmp_for_merge/merged_results_500.csv
# remove the top line
sed -i '1d' tmp_for_merge/merged_results_500.csv

# sort them by the first column alphabetically
sort -t, -k1,1 tmp_for_merge/merged_results_500.csv > tmp_for_merge/merged_results_500_sorted.csv
mv tmp_for_merge/merged_results_500_sorted.csv tmp_for_merge/merged_results_500.csv

# kjore merge_inference_results.py for alle resultater
python3 nibio_inference/merge_inference_results.py -i ~/data/"$folder_name"/results_100/ -o tmp_for_merge/merged_results_100.csv
# remove the top line
sed -i '1d' tmp_for_merge/merged_results_100.csv

# sort them by the first column alphabetically
sort -t, -k1,1 tmp_for_merge/merged_results_100.csv > tmp_for_merge/merged_results_100_sorted.csv
mv tmp_for_merge/merged_results_100_sorted.csv tmp_for_merge/merged_results_100.csv

# kjore merge_inference_results.py for alle resultater
python3 nibio_inference/merge_inference_results.py -i ~/data/"$folder_name"/results_10/ -o tmp_for_merge/merged_results_10.csv
# remove the top line
sed -i '1d' tmp_for_merge/merged_results_10.csv

# sort them by the first column alphabetically
sort -t, -k1,1 tmp_for_merge/merged_results_10.csv > tmp_for_merge/merged_results_10_sorted.csv
mv tmp_for_merge/merged_results_10_sorted.csv tmp_for_merge/merged_results_10.csv


# kjore cat for alle resultater, begynner med org
cat tmp_for_merge/merged_results_org.csv tmp_for_merge/merged_results_1000.csv tmp_for_merge/merged_results_500.csv tmp_for_merge/merged_results_100.csv tmp_for_merge/merged_results_10.csv > tmp_for_merge/merged_results.csv