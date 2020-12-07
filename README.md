# muban


####请运行Measure.py

input: Color image

output: Measurement renderings images & print messages 

samples中包含部分测试图, 可通过修改img_path输入一张裁剪出目标木堆的color图像的路径

##
当前假设及解决办法：
1. 彩色图已经crop出目标木堆 ；          可用深度图去除该假设
2. 拍摄的左右角度不能有较大倾斜（大于10度）；  可用深度图辅助做彩色图的warp
3. 顶层未发生严重弯曲
4. 非常依赖底、定层两条边的正确拟合   
