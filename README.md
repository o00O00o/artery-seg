四类分别是：背景，血管内径，钙化斑块，软斑块  

四类的slice个数：[125766. 125763.   4146.    828.]

先按一个比例分好，在两份数据里分别做过采样

统计记录整个数据集软硬板块的位置
保存成csv文件，一行记录一个slice的软硬斑块的位置   格式为：case_id, branch_id, slice_id


unlabel: [3, 7, 8, 9, 10, 11, 12, 13, 14, 18, 19, 20, 21, 28, 29, 30, 36, 37, 39, 40, 41, 42, 44, 46, 47, 48, 49, 50, 51, 52, 53, 58, 59, 60, 65, 68, 71, 72, 73, 74, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 90, 91, 92, 93, 95, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 111, 112, 113, 114, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126]
label: [127, 128, 129, 130, 131, 132, 133, 134, 135, 137]
test: [138, 139, 140, 141, 142, 143, 144, 146, 147, 148, 149]
num: 3357, 487, 917

所以 unlabel: 0-124     label: 124-140    test: 140-149

effective_num: 3325, 583, 853

total_num: 102637, 14513, 8616

augmented: 