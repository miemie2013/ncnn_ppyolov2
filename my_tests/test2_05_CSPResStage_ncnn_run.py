
'''
（在pycharm的普通终端输入）：
cd build/examples
.\test2_01_conv_ncnn ../../my_tests/my_test.jpg ../../my_tests/05.param ../../my_tests/05.bin

(linux)
cd build/examples
./test2_01_conv_ncnn ../../my_tests/my_test.jpg ../../my_tests/05.param ../../my_tests/05.bin

cd build/examples
./test2_01_conv_ncnn ../../my_tests/my_test.jpg ../../my_tests/05_pncnn.param ../../my_tests/05_pncnn.bin


(比较输出的不同)
cd my_tests
python diff_output.py --ncnn_output ../build/examples/output.txt --torch_output 05.npz


'''



