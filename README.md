# XiaoXu-的学习笔记
## 自我介绍
我是SLAM练习时长一年半的练习生小徐不会唱、跳、rap和打篮球，这个project用来记录自己SLAM的学习过程。
## 2022/5/12之前的工作
1. 本科毕业设计，基于主动视觉的移动机器人路径规划与导航[实物](https://www.bilibili.com/video/BV1UA411g7Lu?spm_id_from=333.999.0.0),[仿真](https://www.bilibili.com/video/BV1gB4y1u7my?spm_id_from=333.999.0.0)
2. 柑橘采摘机器人，主要负责Qt[可视化界面](https://github.com/LeiXu1999/SLAM-Learning-Notes/tree/main/%E5%AD%A6%E4%B9%A0%E8%AE%B0%E5%BD%95/%E6%9F%91%E6%A9%98%E9%87%87%E6%91%98%E5%8F%AF%E8%A7%86%E5%8C%96%E7%95%8C%E9%9D%A2)和各模块的交互。
## 2022/5/12
1. 初入Github，学习[markdown](https://github.com/LeiXu1999/XiaoXu-/blob/main/%E5%AD%A6%E4%B9%A0%E8%AE%B0%E5%BD%95/Markdown.md)。
2. ALOAM用kitti数据集计算pose，然后打印出时间戳转成tum用evo工具评价。
3. 学习如何[管理github的仓库](https://www.bilibili.com/video/BV1Vh41187ik?spm_id_from=333.1007.top_right_bar_window_default_collection.content.click)，[笔记](https://github.com/LeiXu1999/SLAM-Learning-Notes/blob/main/%E5%AD%A6%E4%B9%A0%E8%AE%B0%E5%BD%95/git%E7%AE%A1%E7%90%86github%E7%AE%80%E4%BB%8B.md)。
## 2022/5/13~14
1. 整理ALOAM的kitti转tum的package并[上传github](https://github.com/LeiXu1999/A-LOAM-for-kitti-dataset-to-tum.git)。
## 2022/5/15~16
1. 上午：最近邻问题（二叉树、kd输、八叉树）
2. 下午：测试篮球场的LOAM[记录](https://github.com/LeiXu1999/SLAM-Learning-Notes/tree/main/%E5%AD%A6%E4%B9%A0%E8%AE%B0%E5%BD%95/ALOAM%20%E5%AE%9E%E9%AA%8C%E8%AE%B0%E5%BD%95/2022_05_16%E5%85%AB%E5%85%AC%E5%AF%93%E5%92%8C%E7%AF%AE%E7%90%83%E5%9C%BA)，漂移很严重。
## 2022/5/17
1. 上午：bag of words 词袋模型 
2. 下午：线特征
## 2022/5/18
1. 上午：录bag测线特征提取，发现LSD提线断点很严重。一条直线可能会变成几个小线段，并且光源对纹理丰富地方变换对线的提取有很大影响（应该设定一些阈值）
2. 下午：多机器人路径规划的论文
## 2022/5/19
1. 上午：MLE和Fisher information还未来得及整理
2. 下午：ALOAM测试校园后山[记录](https://github.com/LeiXu1999/SLAM-Learning-Notes/tree/main/%E5%AD%A6%E4%B9%A0%E8%AE%B0%E5%BD%95/ALOAM%20%E5%AE%9E%E9%AA%8C%E8%AE%B0%E5%BD%95/2022_05_18%E5%90%8E%E5%B1%B1/%E5%90%8E%E5%B1%B1)，没有回环在稠密的地方不怎么漂太强了。
## 2022/5/20
1. 上午：找了一个Fisher information 相关的包 ETH的，但是没跑起来，不打算跑了，用到再说。
2. 下午：整理一下这周学的东西。
## 2022/5/21~22
1. 考虑开题的事（摸鱼:)）
## 2022/5/23 开始考虑LeetCode
1. 上午：ORB提取与匹配的方法
2. 下午：线特征代码
3. 晚上: LeetCode：**做项目要保持对难点的敏感程度，记录难点的解决过程**
## 2022/5/24
1. 上午：跟导师确定硕士课题
2. 下午：笔记本双系统CUDA
3. 晚上: 装ZED SDK
## 2022/5/25
1. 上午：整理了一下基础知识，过几天放出来
2. 下午：装一下ZED的ROS包 为后续录包做准备
3. 晚上: LeetCode：时间复杂度，空间复杂度，704.二分查找（有序无重复）
## 2022/5/26~27
1. 上午：[ORB源码ICP、PnP、三角化、F、E、H矩阵](https://github.com/LeiXu1999/SLAM-Learning-Notes/blob/main/%E5%AD%A6%E4%B9%A0%E8%AE%B0%E5%BD%95/ORB%E7%AC%94%E8%AE%B0/ORBSLAM2%E9%A2%84%E5%A4%87%E7%9F%A5%E8%AF%86%E4%B9%8B%E8%BF%90%E5%8A%A8%E5%92%8C%E8%A7%82%E6%B5%8B%E7%9A%84%E6%B1%82%E8%A7%A3(ICP%20PnP%20%E4%B8%89%E8%A7%92%E5%8C%96%20F%20H).md) 终于是推完了也清楚怎么用了，搞得我蜕了一层皮
2. 下午：线特征
3. 晚上：LeetCode：27.移除元素（双指针法，快指针++慢指针非移除元素的时候++）209.长度最小的子数组（滑动窗口，快指针++慢指针满足条件++并减窗口左数据）
