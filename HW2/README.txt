## Team

Garima Gupta: ggupta22@uic.edu
Sai Teja Karnati: skarna3@uic.edu
Shubham Singh: ssing57@uic.edu
Wangfei Wang: wwang75@uic.edu


## Dependencies

A "requirements.txt" file is provided to install any dependencies needed to run the code.

The code is organized as per the assignment questions:

## Q3a
`code/conv_test.py` contains the code to initialise the kernel, and use our implementation on 2D convolution in `conv.py` to print the result as standard output. We also cross-check our implementation with PyTorch's implementation on convolution function in `torch.nn.functional.conv2d`.

## Q4

###(b)
The code for this part can be found in `code/train.py`. It uses `crf.py` to initialise the model and print the training accuracies as standard output. It can be run as:
```
$ python code/train.py
```

###(c)
We created new scripts for this part, namely `code/train_4c.py` and `crf_4c.py`. The major difference is adding a new convolution layer. It can be run as:
```
$ python code/train_4c.py
```

###(d)
The code for part (b) and (c) is device-agnostic, so it should run fine on CPU and GPU machines. All our plots were made using `plot.py`. Please see the report for wallclock time analysis.


## Q5
The code for this part is organized in a new directory `Q5`. Please refer to `/code/Q5/readme.txt` for details.