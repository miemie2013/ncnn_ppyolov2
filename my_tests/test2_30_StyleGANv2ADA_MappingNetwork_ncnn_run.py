
'''
（在pycharm的普通终端输入）：
cd build/examples
.\test2_08_stylegan_ncnn ../../my_tests/seed_75.bin ../../my_tests/30_pncnn.param ../../my_tests/30_pncnn.bin 1

(linux)
cd build/examples
./test2_08_stylegan_ncnn ../../my_tests/seed_75.bin ../../my_tests/30_pncnn.param ../../my_tests/30_pncnn.bin 1


(比较输出的不同)
cd my_tests
python diff_output.py --ncnn_output ../build/examples/output.txt --torch_output 30.npz


'''



