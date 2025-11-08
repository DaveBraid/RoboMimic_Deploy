import numpy as np

mj = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
lab = np.zeros_like(mj)

mj2lab = [0, 2, 4, 6, 8, 1, 3, 5, 7]

lab = mj[mj2lab]

print("MJ:", mj)
print("Lab:", lab)

'''
所以 mj2lab 这个映射表可以理解为：
将对应位置的值，替换成该位置index对应的值。

例如：
- 0号位置用0号元素1
- 1号位置用2号元素3

最终会将使用这个映射表的数组元素的顺序调整成映射表中指定的顺序。
得到的结果是size一样的，与要求顺序一致的数组(lab)
'''

mj_re = np.zeros_like(mj)
mj_re[mj2lab] = lab

print("Reconstructed MJ:", mj_re)
