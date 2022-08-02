
'''
（在pycharm的普通终端输入）：
cd build/examples
.\test2_07_lrelu_ncnn ../../my_tests/my_test.jpg ../../my_tests/09.param ../../my_tests/09.bin

(linux)
cd build/examples
./test2_07_lrelu_ncnn ../../my_tests/my_test.jpg ../../my_tests/09.param ../../my_tests/09.bin

cd build/examples
./test2_07_lrelu_ncnn ../../my_tests/my_test.jpg ../../my_tests/09_pncnn.param ../../my_tests/09_pncnn.bin


(比较输出的不同)
cd my_tests
python diff_output.py --ncnn_output ../build/examples/output.txt --torch_output 09.npz


'''



