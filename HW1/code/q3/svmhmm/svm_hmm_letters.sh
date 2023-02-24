#!/bin/bash

values='1 10 100 1000 5000'
for value in $values
do
	./svm_hmm_learn -c $value train_struct.txt msl$value.dat
    ./svm_hmm_classify test_struct.txt msl$value.dat p_labels$value.txt >>resultoutput.txt
done
echo All done
