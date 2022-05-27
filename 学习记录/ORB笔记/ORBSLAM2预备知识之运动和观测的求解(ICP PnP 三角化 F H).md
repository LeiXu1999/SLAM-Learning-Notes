## 一、预备知识点
1. 3D-3D ICP与SE3/Sim3
2. 3D-2D PnP
3. 2D-3D Triangulation
4. 2D-2D Homography
#
### 1. 3D-3D ICP 与 SE3/Sim3
#### （1）3点法求R
三对点，
- 第一帧的三个3D点以一个点为原点另一个点为x方向建立一个正交坐标系得到R_w2o
- 第二帧得到R_w2o'
- 可以得到R_o2o'
#### （2）ICP（Iterative Closest Point）
最小化两帧之间同一个三维点误差：
$$R^*,t^*=argmin\frac{1}{2}\Sigma||P_i-(RP_i'+t)||_2^2$$
#### （3）SVD求解步骤
**对去中心化的两组点q’×q置求和算SVD， R为VU置，t从去中心化点倒推**
- 输入两组匹配的三维点
- 求每组三维点的中心点
- 算这两组去中心化的三维点
- $$\Sigma q_i'q_i^T=U\Sigma V^T$$
- $$ R^*=VU^T$$
- $$t=P_均-RP_均'$$

**ORB 中不是这个而且求的最优四元数，其实就是四元数和旋转矩阵之间转化算出$$\Sigma q_i'q_i^T$$然后变换一下对其SVD后变成最大奇异值对应的特征向量就是最优四元数在罗德里格斯公式转成R就行了**
#### （4）推导具体看笔记
- 对误差函数ATA的展开
- 去中心化处理，可以保证先求最优R然后求最优t
- 调出只于R有关的量
- 正交性质：旋转矩阵的正交性 保证RTR为E
- 因为最优化后面的目标函数是一个1*1的数字，可以转化成矩阵的迹
- 迹的性质：乘法交换律和加法结合律，tr（AB）=tr（BA）tr（A+B）=tr（A）+tr（B）
- 先用加法律把迹放进去 再用乘法把qRq的R放左边然后再用加法把迹和R弄到求和左边
- 这是最大化这个东西实际上是保证R成求和q‘q是最大的
- 用迹的schwarz不等式ab<=（ata*btb）^0.5
- 对求和q'q进行SVD再用迹的性质把奇异值矩阵放到最右边，其他的都是正交阵，因此正交阵的迹最高是E即满足ATA的形式这时就有R bast了
- 然后倒推求t best

代码：

**在ORB中Sim3 主要求解回环的问题，判断回环以后 算R和t**

**ORBSLAM思想是通过ICP算出Rt来之后返回对应的2D图像进行验证（RANSAC），如果不满足liner就不要这次的R、t；**
```C++
Sim3Solver.cpp
/*****************************************************************************/
//1. 准备工作
//取3D的配对点mvpMapPoints和序号mvnIndices，
//原Rt结算3D对应的2D的配对点mvP1im，
//2D重投影误差阈值mvnMaxError，
//取出3D点的序号mvAllIndices（防止RANSAC重复使用），
//RANSAC参数复位
Sim3Solver::Sim3Solver(KeyFrame *pKF1, KeyFrame *pKF2, const vector<MapPoint *> &vpMatched12, const bool bFixScale):
    mnIterations(0), mnBestInliers(0), mbFixScale(bFixScale)
{
    mpKF1 = pKF1;
    mpKF2 = pKF2;

    vector<MapPoint*> vpKeyFrameMP1 = pKF1->GetMapPointMatches();
    mN1 = vpMatched12.size();

    mvpMapPoints1.reserve(mN1);
    mvpMapPoints2.reserve(mN1);
    mvpMatches12 = vpMatched12;
    mvnIndices1.reserve(mN1);
    mvX3Dc1.reserve(mN1);
    mvX3Dc2.reserve(mN1);

    cv::Mat Rcw1 = pKF1->GetRotation();
    cv::Mat tcw1 = pKF1->GetTranslation();
    cv::Mat Rcw2 = pKF2->GetRotation();
    cv::Mat tcw2 = pKF2->GetTranslation();

    mvAllIndices.reserve(mN1);

    size_t idx=0;
    // mN1为pKF2特征点的个数
    for(int i1=0; i1<mN1; i1++)
    {
        // 如果该特征点在pKF1中有匹配
        if(vpMatched12[i1])
        {
            // step1: 根据vpMatched12配对比配的MapPoint： pMP1和pMP2
            MapPoint* pMP1 = vpKeyFrameMP1[i1];
            MapPoint* pMP2 = vpMatched12[i1];

            if(!pMP1)
                continue;

            if(pMP1->isBad() || pMP2->isBad())
                continue;

            // step2：计算允许的重投影误差阈值：mvnMaxError1和mvnMaxError2
            // 注：是相对当前位姿投影3D点得到的图像坐标，见step6
            // step2.1：根据匹配的MapPoint找到对应匹配特征点的索引：indexKF1和indexKF2
            int indexKF1 = pMP1->GetIndexInKeyFrame(pKF1);
            int indexKF2 = pMP2->GetIndexInKeyFrame(pKF2);

            if(indexKF1<0 || indexKF2<0)
                continue;

            // step2.2：取出匹配特征点的引用：kp1和kp2
            const cv::KeyPoint &kp1 = pKF1->mvKeysUn[indexKF1];
            const cv::KeyPoint &kp2 = pKF2->mvKeysUn[indexKF2];

            // step2.3：根据特征点的尺度计算对应的误差阈值：mvnMaxError1和mvnMaxError2 octave是金字塔层数mvLevelSigma2缩放因子平方
            const float sigmaSquare1 = pKF1->mvLevelSigma2[kp1.octave];
            const float sigmaSquare2 = pKF2->mvLevelSigma2[kp2.octave];

            mvnMaxError1.push_back(9.210*sigmaSquare1);
            mvnMaxError2.push_back(9.210*sigmaSquare2);

            // mvpMapPoints1和mvpMapPoints2是匹配的MapPoints容器
            mvpMapPoints1.push_back(pMP1);
            mvpMapPoints2.push_back(pMP2);
            mvnIndices1.push_back(i1);

            // step4：将MapPoint从世界坐标系变换到相机坐标系：mvX3Dc1和mvX3Dc2
            cv::Mat X3D1w = pMP1->GetWorldPos();
            mvX3Dc1.push_back(Rcw1*X3D1w+tcw1);

            cv::Mat X3D2w = pMP2->GetWorldPos();
            mvX3Dc2.push_back(Rcw2*X3D2w+tcw2);

            mvAllIndices.push_back(idx);
            idx++;
        }
    }

    // step5：两个关键帧的内参
    mK1 = pKF1->mK;
    mK2 = pKF2->mK;

    // step6：记录计算两针Sim3之前3D mappoint在图像上的投影坐标：mvP1im1和mvP2im2
    FromCameraToImage(mvX3Dc1,mvP1im1,mK1);
    FromCameraToImage(mvX3Dc2,mvP2im2,mK2);

    SetRansacParameters();
}
/****************************************************************************/
//2.RABSAC的操作，前面一堆判定工作
//随机选三个组点算，先出来就把候选数组对应的三组点给pop出去  三对六个点
//将三组点按照Sim3计算（里面没有求和累加 实际上已经转换成矩阵相乘了）q‘q^T 然后按照上述做了转化
//通过投影误差进行inlier检测
//记录inlier最大的点，当inlier大到一定阈值 直接返回最优的T 不refine
//Ransac求解mvX3Dc1和mvX3Dc2之间Sim3，函数返回mvX3Dc2到mvX3Dc1的Sim3变换
cv::Mat Sim3Solver::iterate(int nIterations, bool &bNoMore, vector<bool> &vbInliers, int &nInliers)
{
    bNoMore = false;
    vbInliers = vector<bool>(mN1,false);
    nInliers=0;

    if(N<mRansacMinInliers) //如果 样本综述小于RANSAC最小内点个数则没有必要RANSAC
    {
        bNoMore = true;
        return cv::Mat();
    }

    vector<size_t> vAvailableIndices;

    cv::Mat P3Dc1i(3,3,CV_32F);
    cv::Mat P3Dc2i(3,3,CV_32F);

    int nCurrentIterations = 0;
    while(mnIterations<mRansacMaxIts && nCurrentIterations<nIterations)
    {
        nCurrentIterations++;// 这个函数中迭代的次数
        mnIterations++;// 总的迭代次数，默认为最大为300

        vAvailableIndices = mvAllIndices;

        // Get min set of points
        // 步骤1：任意取三组点算Sim矩阵
        for(short i = 0; i < 3; ++i)
        {
            int randi = DUtils::Random::RandomInt(0, vAvailableIndices.size()-1);

            int idx = vAvailableIndices[randi];

            // P3Dc1i和P3Dc2i中点的排列顺序：
            // x1 x2 x3 ...
            // y1 y2 y3 ...
            // z1 z2 z3 ...
            mvX3Dc1[idx].copyTo(P3Dc1i.col(i));
            mvX3Dc2[idx].copyTo(P3Dc2i.col(i));

            vAvailableIndices[randi] = vAvailableIndices.back();
            vAvailableIndices.pop_back();
        }

        // 步骤2：根据两组匹配的3D点，计算之间的Sim3变换
        ComputeSim3(P3Dc1i,P3Dc2i);

        // 步骤3：通过投影误差进行inlier检测
        CheckInliers();

        if(mnInliersi>=mnBestInliers)
        {
            mvbBestInliers = mvbInliersi;
            mnBestInliers = mnInliersi;
            mBestT12 = mT12i.clone();
            mBestRotation = mR12i.clone();
            mBestTranslation = mt12i.clone();
            mBestScale = ms12i;

            if(mnInliersi>mRansacMinInliers)
            {
                nInliers = mnInliersi;
                for(int i=0; i<N; i++)
                    if(mvbInliersi[i])
                        vbInliers[mvnIndices1[i]] = true;

                // ！！！！！note: 1. 只要计算得到一次合格的Sim变换，就直接返回 2. 没有对所有的inlier进行一次refine操作
                return mBestT12;
            }
        }
    }

    if(mnIterations>=mRansacMaxIts)
        bNoMore=true;

    return cv::Mat();
}
/*****************************************************************/
//3.推到的具体公式的运用
void Sim3Solver::ComputeSim3(cv::Mat &P1, cv::Mat &P2)
{
    // ！！！！！！！这段代码一定要看这篇论文！！！！！！！！！！！
    // Custom implementation of:
    // Horn 1987, Closed-form solution of absolute orientataion using unit quaternions

    // Step 1: Centroid and relative coordinates（模型坐标系）
    cv::Mat Pr1(P1.size(),P1.type()); // Relative coordinates to centroid (set 1)
    cv::Mat Pr2(P2.size(),P2.type()); // Relative coordinates to centroid (set 2)
    cv::Mat O1(3,1,Pr1.type()); // Centroid of P1
    cv::Mat O2(3,1,Pr2.type()); // Centroid of P2

    // O1和O2分别为P1和P2矩阵中3D点的质心
    // Pr1和Pr2为减去质心后的3D点
    ComputeCentroid(P1,Pr1,O1);
    ComputeCentroid(P2,Pr2,O2);

    // Step 2: Compute M matrix
    cv::Mat M = Pr2*Pr1.t();

    // Step 3: Compute N matrix
    double N11, N12, N13, N14, N22, N23, N24, N33, N34, N44;

    cv::Mat N(4,4,P1.type());

    N11 = M.at<float>(0,0)+M.at<float>(1,1)+M.at<float>(2,2);
    N12 = M.at<float>(1,2)-M.at<float>(2,1);
    N13 = M.at<float>(2,0)-M.at<float>(0,2);
    N14 = M.at<float>(0,1)-M.at<float>(1,0);
    N22 = M.at<float>(0,0)-M.at<float>(1,1)-M.at<float>(2,2);
    N23 = M.at<float>(0,1)+M.at<float>(1,0);
    N24 = M.at<float>(2,0)+M.at<float>(0,2);
    N33 = -M.at<float>(0,0)+M.at<float>(1,1)-M.at<float>(2,2);
    N34 = M.at<float>(1,2)+M.at<float>(2,1);
    N44 = -M.at<float>(0,0)-M.at<float>(1,1)+M.at<float>(2,2);

    N = (cv::Mat_<float>(4,4) << N11, N12, N13, N14,
                                 N12, N22, N23, N24,
                                 N13, N23, N33, N34,
                                 N14, N24, N34, N44);


    // Step 4: Eigenvector of the highest eigenvalue
    cv::Mat eval, evec;

    cv::eigen(N,eval,evec); //evec[0] is the quaternion of the desired rotation

    // N矩阵最大特征值（第一个特征值）对应特征向量就是要求的四元数死（q0 q1 q2 q3）
    // 将(q1 q2 q3)放入vec行向量，vec就是四元数旋转轴乘以sin(ang/2)
    cv::Mat vec(1,3,evec.type());
    (evec.row(0).colRange(1,4)).copyTo(vec); //extract imaginary part of the quaternion (sin*axis)

    // Rotation angle. sin is the norm of the imaginary part, cos is the real part
    double ang=atan2(norm(vec),evec.at<float>(0,0));

    vec = 2*ang*vec/norm(vec); //Angle-axis representation. quaternion angle is the half

    mR12i.create(3,3,P1.type());

    cv::Rodrigues(vec,mR12i); // computes the rotation matrix from angle-axis

    // Step 5: Rotate set 2
    cv::Mat P3 = mR12i*Pr2;

    // Step 6: Scale
    if(!mbFixScale)
    {
        // 论文中还有一个求尺度的公式，p632右中的位置，那个公式不用考虑旋转
        double nom = Pr1.dot(P3);
        cv::Mat aux_P3(P3.size(),P3.type());
        aux_P3=P3;
        cv::pow(P3,2,aux_P3);
        double den = 0;

        for(int i=0; i<aux_P3.rows; i++)
        {
            for(int j=0; j<aux_P3.cols; j++)
            {
                den+=aux_P3.at<float>(i,j);
            }
        }

        ms12i = nom/den;
    }
    else
        ms12i = 1.0f;

    // Step 7: Translation
    mt12i.create(1,3,P1.type());
    mt12i = O1 - ms12i*mR12i*O2;

    // Step 8: Transformation
    // Step 8.1 T12
    mT12i = cv::Mat::eye(4,4,P1.type());

    cv::Mat sR = ms12i*mR12i;

    //         |sR t|
    // mT12i = | 0 1|
    sR.copyTo(mT12i.rowRange(0,3).colRange(0,3));
    mt12i.copyTo(mT12i.rowRange(0,3).col(3));

    // Step 8.2 T21
    mT21i = cv::Mat::eye(4,4,P1.type());

    cv::Mat sRinv = (1.0/ms12i)*mR12i.t();

    sRinv.copyTo(mT21i.rowRange(0,3).colRange(0,3));
    cv::Mat tinv = -sRinv*mt12i;
    tinv.copyTo(mT21i.rowRange(0,3).col(3));
}
/********************************************************/
// 检测算出的新Rt满足内点的个数
//2系的3D点通过新的Rt变换的1系的相机坐标系下，1系同理
//先算距离向量相减，然后相减的误差进行点成，A.dot(A)欧氏距离
//判断距离是否满足 MaxError，计数
void Sim3Solver::CheckInliers()
{
    vector<cv::Mat> vP1im2, vP2im1;
    Project(mvX3Dc2,vP2im1,mT12i,mK1);// 把2系中的3D经过Sim3变换(mT12i)到1系中计算重投影坐标
    Project(mvX3Dc1,vP1im2,mT21i,mK2);// 把1系中的3D经过Sim3变换(mT21i)到2系中计算重投影坐标

    mnInliersi=0;

    for(size_t i=0; i<mvP1im1.size(); i++)
    {
        cv::Mat dist1 = mvP1im1[i]-vP2im1[i];
        cv::Mat dist2 = vP1im2[i]-mvP2im2[i];

        const float err1 = dist1.dot(dist1);
        const float err2 = dist2.dot(dist2);

        if(err1<mvnMaxError1[i] && err2<mvnMaxError2[i])
        {
            mvbInliersi[i]=true;
            mnInliersi++;
        }
        else
            mvbInliersi[i]=false;
    }
}

```
#
### 2. 3D-2D PnP
直接法：P3P/DLT/EPnP优化法：LHM/BA
#### （1）P3P 需要相机内参
2D点通过内参变成3D点，然后3D-3D 三点法求R
#### （2）DLT 6个点分解K、R、T
- 凡是x1=/尺度Mx2，x1叉乘x2=0的形式式子我们都可以转化成 Ax=0 来求解就叫做DLT直接线性变换
- 具体怎么求Ax=0，就是一个线性最小二乘问题
- 线性最小二乘的SVD分解，就是最小奇异值的右奇异向量
##### 推导
$$A^*=argmin||Ax||_2^2$$
直接对A进行SVD分解然后展开就能推出来
#### （3）EPnP最小4个点
- 12个未知数，用四对点解出来的没有尺度（8个式子），真实的解应该是 求出来的特征向量的线性叠加，通过添加几何约束来求解。摄像机坐标系控制点之间的距离等于在世界坐标系中计算得到的距离
- EPnP在check inliers用到了相机内参（依赖相机内参）
- 构造控制点，去中心化，得到中心点做C0
- 然后PCL（SVD）求三个主方向归一化得到C1C2C3
- 每个点用控制点C来表示 并能求出四个系数α相加为1
- 可以保证世界坐标系和相机坐标系的系数一致性，即求出了相机坐标系下面的系数
- 现在目标就是知道系数求解相机坐标系四个控制点
- 用内参构建相机模型 即在系数已知和2D坐标的情况下列相机模型求相机坐标系控制点（联系到2D点了）
- 求出相机坐标系下的控制点就能结算w2c的R了
- 怎么求 相机模型转换成DLT问题然后SVD分解 求出了四个特征向量 对四个向量加权 后面就是密码学的知识了，核心思想是16带求变量，假设几个变量已知或者相等，先解算出几组解，做初解然后放到GN里面迭代

代码：

**EPnP在ORB-SLAM中主要用于Tracking线程中的重定位Relocalization模块，须要经过当前关键帧Bow与候选帧匹配上的3D地图点，迅速创建当前相机的初始姿态。Epnp是基于一个标定好的针孔相机模型的，ORB3采用独立于相机的ML-pnp算法可以完全从相机模型中解耦，因为他利用投影光线作为输入。相机模型只需要提供一个从像素传递到投影光线的反投影函数，以便能够使用重定位。**

**首先通过BOW进行特征匹配，选出候选关键帧（在ORB-SLAM中所有关键帧都会被记录到BOW数据库中）如果匹配数目超过15个则进行EPnP求解，通过5次迭代RNASAC得到相机初始值，然后进行优化相机位姿。如果优化后匹配点小于50，则通过投影对之前未匹配的点进行再次计算。**
```C++
PnPsolver.cpp
//通过词袋模型检测当前帧和候选帧之间的匹配程度，如果很匹配，就用当前帧的2D点与候选帧建出来的3D点进行PnP
/***************************************************************************************/
//1. 准备工作，
//记录3D点mvP3Dw、
//2D点mvP2D和
//索引列表mvKeyPointIndices
//RANSAC序号mvAllIndices


// pcs表示3D点在camera坐标系下的坐标
// pws表示3D点在世界坐标系下的坐标
// us表示图像坐标系下的2D点坐标
// alphas为真实3D点用4个虚拟控制点表达时的系数
PnPsolver::PnPsolver(const Frame &F, const vector<MapPoint*> &vpMapPointMatches):
    pws(0), us(0), alphas(0), pcs(0), maximum_number_of_correspondences(0), number_of_correspondences(0), mnInliersi(0),
    mnIterations(0), mnBestInliers(0), N(0)
{
    // mnIterations记录当前已经Ransac的次数
    // pws 每个3D点有(X Y Z)三个值
    // us 每个图像2D点有(u v)两个值
    // alphas 每个3D点由四个控制点拟合，有四个系数
    // pcs 每个3D点有(X Y Z)三个值
    // maximum_number_of_correspondences 用于确定当前迭代所需内存是否够用

    // 根据点数初始化容器的大小
    mvpMapPointMatches = vpMapPointMatches;
    mvP2D.reserve(F.mvpMapPoints.size());
    mvSigma2.reserve(F.mvpMapPoints.size());
    mvP3Dw.reserve(F.mvpMapPoints.size());
    mvKeyPointIndices.reserve(F.mvpMapPoints.size());
    mvAllIndices.reserve(F.mvpMapPoints.size());

    int idx=0;
    for(size_t i=0, iend=vpMapPointMatches.size(); i<iend; i++)
    {
        MapPoint* pMP = vpMapPointMatches[i];//依次获取一个MapPoint

        if(pMP)
        {
            if(!pMP->isBad())
            {
                const cv::KeyPoint &kp = F.mvKeysUn[i];//得到2维特征点, 将KeyPoint类型变为Point2f

                mvP2D.push_back(kp.pt);//存放到mvP2D容器
                mvSigma2.push_back(F.mvLevelSigma2[kp.octave]);//记录特征点是在哪一层提取出来的

                cv::Mat Pos = pMP->GetWorldPos();//世界坐标系下的3D点
                mvP3Dw.push_back(cv::Point3f(Pos.at<float>(0),Pos.at<float>(1), Pos.at<float>(2)));

                mvKeyPointIndices.push_back(i);//记录被使用特征点在原始特征点容器中的索引, mvKeyPointIndices是跳跃的
                mvAllIndices.push_back(idx);//记录被使用特征点的索引, mvAllIndices是连续的

                idx++;
            }
        }
    }

    // Set camera calibration parameters
    fu = F.fx;
    fv = F.fy;
    uc = F.cx;
    vc = F.cy;

    SetRansacParameters();
}
/***************************************************************************************/
//2.RANSAC
//与 PnP 一样 但是这个有Refine

//选点（默认四对点 用EPnP）
//计算 EPnP 最终也是从三组解中选出最小重投影误差的Rt 
//检测Inliers 算出Rt，将3D点投影到2D算重投影，
//Refine
cv::Mat PnPsolver::iterate(int nIterations, bool &bNoMore, vector<bool> &vbInliers, int &nInliers)
{
    bNoMore = false;
    vbInliers.clear();
    nInliers=0;

    // mRansacMinSet为每次RANSAC需要的特征点数，默认为4组3D-2D对应点
    set_maximum_number_of_correspondences(mRansacMinSet);

    // N为所有2D点的个数, mRansacMinInliers为RANSAC迭代终止的inlier阈值，如果已经大于所有点的个数，则停止迭代
    if(N<mRansacMinInliers)
    {
        bNoMore = true;
        return cv::Mat();
    }

    // mvAllIndices为所有参与PnP的2D点的索引
    // 每次ransac，将mvAllIndices赋值一份给vAvailableIndices，并从中随机挑选mRansacMinSet组3D-2D对应点进行一次RANSAC
    vector<size_t> vAvailableIndices;

    // nIterations: 根据ransac概率值计算出来的迭代次数
    // nCurrentIterations：记录每调用一次iterate函数，内部会有多次迭代
    // mnIterations：记录iterate调用次数 * 内部迭代次数总迭代次数
    int nCurrentIterations = 0;
    while(mnIterations<mRansacMaxIts || nCurrentIterations<nIterations)
    {
        nCurrentIterations++;
        mnIterations++;
        reset_correspondences();

        // 这个赋值稍微有些低效
        // 每次迭代将所有的特征匹配复制一份mvAllIndices--->vAvailableIndices，然后选取mRansacMinSet个进行求解
        vAvailableIndices = mvAllIndices;

        // Get min set of points
        for(short i = 0; i < mRansacMinSet; ++i)
        {
            int randi = DUtils::Random::RandomInt(0, vAvailableIndices.size()-1);

            int idx = vAvailableIndices[randi];

            // 将对应的3D-2D压入到pws和us
            add_correspondence(mvP3Dw[idx].x,mvP3Dw[idx].y,mvP3Dw[idx].z,mvP2D[idx].x,mvP2D[idx].y);

            // ！！！将已经被选中参与ransac的点去除（用vector最后一个点覆盖），避免抽取同一个数据参与ransac
            vAvailableIndices[randi] = vAvailableIndices.back();
            vAvailableIndices.pop_back();
        }

        // Compute camera pose
        // EPnP算法求解R，t
        compute_pose(mRi, mti);

        // Check inliers
        // 统计和记录inlier个数以及符合inlier的点：mnInliersi, mvbInliersi
        CheckInliers();

        if(mnInliersi>=mRansacMinInliers)
        {
            // If it is the best solution so far, save it
            // 记录inlier个数最多的一组解
            if(mnInliersi>mnBestInliers)
            {
                mvbBestInliers = mvbInliersi;
                mnBestInliers = mnInliersi;

                cv::Mat Rcw(3,3,CV_64F,mRi);
                cv::Mat tcw(3,1,CV_64F,mti);
                Rcw.convertTo(Rcw,CV_32F);
                tcw.convertTo(tcw,CV_32F);
                mBestTcw = cv::Mat::eye(4,4,CV_32F);
                Rcw.copyTo(mBestTcw.rowRange(0,3).colRange(0,3));
                tcw.copyTo(mBestTcw.rowRange(0,3).col(3));
            }

            // 将所有符合inlier的3D-2D匹配点一起计算PnP求解R, t
            if(Refine())
            {
                nInliers = mnRefinedInliers;
                vbInliers = vector<bool>(mvpMapPointMatches.size(),false);
                for(int i=0; i<N; i++)
                {
                    if(mvbRefinedInliers[i])
                        vbInliers[mvKeyPointIndices[i]] = true;
                }
                return mRefinedTcw.clone();
            }

        }
    }

    if(mnIterations>=mRansacMaxIts)
    {
        bNoMore=true;
        if(mnBestInliers>=mRansacMinInliers)
        {
            nInliers=mnBestInliers;
            vbInliers = vector<bool>(mvpMapPointMatches.size(),false);
            for(int i=0; i<N; i++)
            {
                if(mvbBestInliers[i])
                    vbInliers[mvKeyPointIndices[i]] = true;
            }
            return mBestTcw.clone();
        }
    }

    return cv::Mat();
}
/*************************************************************************************/
//3. EPnP的计算 计算很复杂，
//算控制点
//算系数
//构造矩阵算相机坐标系的控制点
//约束不够，解应该是特征向量的线性组合
//假设一些变量为0粗略算一个初值 然后高斯牛顿迭代 算了三组解 约束是相机坐标系控制点的尺度和世界坐标系的控制点尺度相同
//三组解 是迭代算出来的相机坐标系下的控制点，通过compute_R_and_t 将Rt算出来然后通过ICP进行重投影误差的计算
//三组解的误差选最小的一个
double PnPsolver::compute_pose(double R[3][3], double t[3])
{
  // 步骤1：获得EPnP算法中的四个控制点（构成质心坐标系）
  choose_control_points();
  // 步骤2：计算世界坐标系下每个3D点用4个控制点线性表达时的系数alphas，公式1
  compute_barycentric_coordinates();

  // 步骤3：构造M矩阵，公式(3)(4)-->(5)(6)(7)
  CvMat * M = cvCreateMat(2 * number_of_correspondences, 12, CV_64F);

  for(int i = 0; i < number_of_correspondences; i++)
    fill_M(M, 2 * i, alphas + 4 * i, us[2 * i], us[2 * i + 1]);

  double mtm[12 * 12], d[12], ut[12 * 12];
  CvMat MtM = cvMat(12, 12, CV_64F, mtm);
  CvMat D   = cvMat(12,  1, CV_64F, d);
  CvMat Ut  = cvMat(12, 12, CV_64F, ut);

  // 步骤3：求解Mx = 0
  // SVD分解M'M
  cvMulTransposed(M, &MtM, 1);
  // 通过（svd分解）求解齐次最小二乘解得到相机坐标系下四个不带尺度的控制点：ut
  // ut的每一行对应一组可能的解
  // 最小特征值对应的特征向量最接近待求的解，由于噪声和约束不足的问题，导致真正的解可能是多个特征向量的线性叠加
  cvSVD(&MtM, &D, &Ut, 0, CV_SVD_MODIFY_A | CV_SVD_U_T);//得到向量ut
  cvReleaseMat(&M);

  // 上述通过求解齐次最小二乘获得解不具有尺度，这里通过构造另外一个最小二乘（L*Betas = Rho）来求解尺度Betas
  // L_6x10 * Betas10x1 = Rho_6x1
  double l_6x10[6 * 10], rho[6];
  CvMat L_6x10 = cvMat(6, 10, CV_64F, l_6x10);
  CvMat Rho    = cvMat(6,  1, CV_64F, rho);

  // Betas10        = [B00 B01 B11 B02 B12 B22 B03 B13 B23 B33]
  // |dv00, 2*dv01, dv11, 2*dv02, 2*dv12, dv22, 2*dv03, 2*dv13, 2*dv23, dv33|, 1*10
  // 4个控制点之间总共有6个距离，因此为6*10
  compute_L_6x10(ut, l_6x10);
  compute_rho(rho);

  double Betas[4][4], rep_errors[4];
  double Rs[4][3][3], ts[4][3];

  // 不管什么情况，都假设论文中N=4，并求解部分betas（如果全求解出来会有冲突）
  // 通过优化得到剩下的betas
  // 最后计算R t

  // Betas10        = [B00 B01 B11 B02 B12 B22 B03 B13 B23 B33]
  // betas_approx_1 = [B00 B01     B02         B03]
  // 建模为除B11、B12、B13、B14四个参数外其它参数均为0进行最小二乘求解，求出B0、B1、B2、B3粗略解
  find_betas_approx_1(&L_6x10, &Rho, Betas[1]);
  // 高斯牛顿法优化B0、B1、B2、B3
  gauss_newton(&L_6x10, &Rho, Betas[1]);
  rep_errors[1] = compute_R_and_t(ut, Betas[1], Rs[1], ts[1]);

  // betas_approx_2 = [B00 B01 B11                            ]
  // 建模为除B00、B01、B11三个参数外其它参数均为0进行最小二乘求解，求出B0、B1、B2、B3粗略解
  find_betas_approx_2(&L_6x10, &Rho, Betas[2]);
  gauss_newton(&L_6x10, &Rho, Betas[2]);
  rep_errors[2] = compute_R_and_t(ut, Betas[2], Rs[2], ts[2]);

  // betas_approx_3 = [B00 B01 B11 B02 B12                    ]
  // 建模为除B00、B01、B11、B02、B12五个参数外其它参数均为0进行最小二乘求解，求出B0、B1、B2、B3粗略解
  find_betas_approx_3(&L_6x10, &Rho, Betas[3]);
  gauss_newton(&L_6x10, &Rho, Betas[3]);
  rep_errors[3] = compute_R_and_t(ut, Betas[3], Rs[3], ts[3]);

  int N = 1;
  if (rep_errors[2] < rep_errors[1]) N = 2;
  if (rep_errors[3] < rep_errors[N]) N = 3;

  copy_R_and_t(Rs[N], ts[N], R, t);

  return rep_errors[N];
}
/************************************************************/
//3.compute_R_and_t
double PnPsolver::compute_R_and_t(const double * ut, const double * betas,
			     double R[3][3], double t[3])
{
  // 通过Betas和特征向量，得到机体坐标系下四个控制点
  compute_ccs(betas, ut);
  // 根据控制点和相机坐标系下每个点与控制点之间的关系，恢复出所有3D点在相机坐标系下的坐标
  compute_pcs();

  // 随便取一个相机坐标系下3D点，如果z < 0，则表明3D点都在相机后面，则3D点坐标整体取负号
  solve_for_sign();

  // 3D-3D svd方法求解ICP获得R，t
  estimate_R_and_t(R, t);

  // 获得R，t后计算所有3D点的重投影误差平均值
  return reprojection_error(R, t);
}
/************************************************************/
// 4.检测内点个数
// 通过之前求解的(R t)检查哪些3D-2D点对属于inliers
// 这里依赖了相机内参
void PnPsolver::CheckInliers()
{
    mnInliersi=0;

    for(int i=0; i<N; i++)
    {
        cv::Point3f P3Dw = mvP3Dw[i];
        cv::Point2f P2D = mvP2D[i];

        // 将3D点由世界坐标系旋转到相机坐标系
        float Xc = mRi[0][0]*P3Dw.x+mRi[0][1]*P3Dw.y+mRi[0][2]*P3Dw.z+mti[0];
        float Yc = mRi[1][0]*P3Dw.x+mRi[1][1]*P3Dw.y+mRi[1][2]*P3Dw.z+mti[1];
        float invZc = 1/(mRi[2][0]*P3Dw.x+mRi[2][1]*P3Dw.y+mRi[2][2]*P3Dw.z+mti[2]);

        // 将相机坐标系下的3D进行针孔投影
        double ue = uc + fu * Xc * invZc;
        double ve = vc + fv * Yc * invZc;

        // 计算残差大小
        float distX = P2D.x-ue;
        float distY = P2D.y-ve;

        float error2 = distX*distX+distY*distY;

        if(error2<mvMaxError[i])
        {
            mvbInliersi[i]=true;
            mnInliersi++;
        }
        else
        {
            mvbInliersi[i]=false;
        }
    }
}
/*************************************************************************/
//5.Refine 用Inliers 在算一次 compute_pose 把inlier 提纯
bool PnPsolver::Refine()
{
    vector<int> vIndices;
    vIndices.reserve(mvbBestInliers.size());

    for(size_t i=0; i<mvbBestInliers.size(); i++)
    {
        if(mvbBestInliers[i])
        {
            vIndices.push_back(i);
        }
    }

    set_maximum_number_of_correspondences(vIndices.size());

    reset_correspondences();

    for(size_t i=0; i<vIndices.size(); i++)
    {
        int idx = vIndices[i];
        add_correspondence(mvP3Dw[idx].x,mvP3Dw[idx].y,mvP3Dw[idx].z,mvP2D[idx].x,mvP2D[idx].y);
    }

    // Compute camera pose
    compute_pose(mRi, mti);

    // Check inliers
    CheckInliers();

    // 通过CheckInliers函数得到那些inlier点用来提纯
    mnRefinedInliers =mnInliersi;
    mvbRefinedInliers = mvbInliersi;

    if(mnInliersi>mRansacMinInliers)
    {
        cv::Mat Rcw(3,3,CV_64F,mRi);
        cv::Mat tcw(3,1,CV_64F,mti);
        Rcw.convertTo(Rcw,CV_32F);
        tcw.convertTo(tcw,CV_32F);
        mRefinedTcw = cv::Mat::eye(4,4,CV_32F);
        Rcw.copyTo(mRefinedTcw.rowRange(0,3).colRange(0,3));
        tcw.copyTo(mRefinedTcw.rowRange(0,3).col(3));
        return true;
    }

    return false;
}
```
#
### 3. 2D-3D 三角化
**初始化的时候会用 比如说 计算F的时候结算出来Rt 我们需要check Rt的四组解 需要通过2D和Rt回复3D点**
$$  \begin{bmatrix}
   u_x  \\
   u_y  \\
   1 
  \end{bmatrix}=\lambda K
  \begin{bmatrix}
   R & t  \\
  \end{bmatrix}
  \begin{bmatrix}
   x  \\
   y  \\
   z  \\
   1  \\
  \end{bmatrix}=\lambda 
  \begin{bmatrix}
   p_1 & p_2 & p_3 & p_4 \\
   p_5 & p_6 & p_7 & p_8 \\
   p_9 & p_{10} & p_{11} & p_{12} 
  \end{bmatrix}
  \begin{bmatrix}
   x  \\
   y  \\
   z  \\
   1  \\
  \end{bmatrix}=\lambda
  \begin{bmatrix}
   P_0^T  \\
   P_1^T  \\
   P_2^T 
   \end{bmatrix}x
  $$
DLT:
$$  \begin{bmatrix}
   u_x  \\
   u_y  \\
   1 
  \end{bmatrix}=\lambda
  \begin{bmatrix}
   P_0^T  \\
   P_1^T  \\
   P_2^T 
   \end{bmatrix}x
  $$
  拆开：
  $$  
   u_x=\lambda P_0^Tx 
  $$
  $$  
   u_y=\lambda P_1^Tx 
  $$
  $$  
   1=\lambda P_2^Tx 
  $$
  化简：
  $$  
   \begin{bmatrix}
   u_x P_2^T-P_0^T  \\
   u_y P_2^T-P_1^T \\
   \end{bmatrix}x=
   \begin{bmatrix}
   0  \\
   0 \\
   \end{bmatrix}
  $$
  
  一对点：
  $$  
   \begin{bmatrix}
   u_x P_2^T-P_0^T  \\
   u_y P_2^T-P_1^T \\
   u_x' P_2^T-P_0^T  \\
   u_y' P_2^T-P_1^T \\
   \end{bmatrix}x=
   \begin{bmatrix}
   0  \\
   0 \\
   0  \\
   0 \\
   \end{bmatrix}
  $$
对A进行SVD分解 x_{best}=最小特征根对应的右特征向量

代码：

**初始化的时候需要三角化来回复尺度**

```C++
void Initializer::Triangulate(const cv::KeyPoint &kp1, const cv::KeyPoint &kp2, const cv::Mat &P1, const cv::Mat &P2, cv::Mat &x3D)
{
    // 在DecomposeE函数和ReconstructH函数中对t有归一化
    // 这里三角化过程中恢复的3D点深度取决于 t 的尺度，
    // 但是这里恢复的3D点并没有决定单目整个SLAM过程的尺度
    // 因为CreateInitialMapMonocular函数对3D点深度会缩放，然后反过来对 t 有改变

    cv::Mat A(4,4,CV_32F);

    A.row(0) = kp1.pt.x*P1.row(2)-P1.row(0);
    A.row(1) = kp1.pt.y*P1.row(2)-P1.row(1);
    A.row(2) = kp2.pt.x*P2.row(2)-P2.row(0);
    A.row(3) = kp2.pt.y*P2.row(2)-P2.row(1);

    cv::Mat u,w,vt;
    cv::SVD::compute(A,w,u,vt,cv::SVD::MODIFY_A| cv::SVD::FULL_UV);
    x3D = vt.row(3).t();
    x3D = x3D.rowRange(0,3)/x3D.at<float>(3);
}
```
#
### 4. 2D-2D对极几何和单应变换


- **对极约束与场景无关，在任何场景结构都成立，对极约束能把点约束在一条极线上**

- **单应假设同试点在一个平面上，就能把点在极线上的确切位置约束住**

- **相机做纯旋转对极几何无解 因为t为0 整个式子无限解**

- **相机做纯旋转单应矩阵还是可以算的**

- **本质矩阵描述了空间中一点对应的两个归一化平面(没有尺度)坐标之间的约束关系。**

- **单应矩阵描述了两个平面之间的映射关系。**

- 对极几何求解本质矩阵E和基础矩阵F
- 单应变换求解H矩阵
- ORB中F矩阵和E矩阵都是归一化坐标之后算的，算出来以后在恢复尺度
- 归一化首先进行了一个平移（移动到点集中心）然后记性了X、y方向上的缩放 跟内参矩阵的形式实际上是一样的，可以理解成投影到归一化平面
- 平移尺度：x的平均 y的平均 ；；缩放尺度x:（1/x方向中心化距离差均值）y:（1/y方向中心化距离差均值）
#### （1）对极几何

**对极约束与场景无关，在任何场景结构都成立，对极约束能把点约束在一条极线上**
$$ s_1u_1 =Kp,s_2u_2=K(Rp+t)$$
$$ s_1K^{-1}u_1 =p,s_2K^{-1}u_2=Rp+t$$
$$ s_2K^{-1}u_2=R s_1K^{-1}u_1+t$$
左乘t的反对称矩阵
$$ t^×s_2K^{-1}u_2=t^×R s_1K^{-1}u_1$$
左乘u2k逆的置
$$ s_2u_2^TK^{-T}t^×K^{-1}u_2=s_1u_2^TK^{-T}t^×RK^{-1} u_1$$
一个向量的反对称矩阵*一个向量等于两个向量叉乘，这个在于原向量相乘=0
$$0=s_1u_2^TK^{-T}t^×RK^{-1} u_1$$
$$0=u_2^TFu_1$$
$$0=u_2^TK^{-T}EK^{-1} u_1$$
1. 本质矩阵：t的反对称矩阵*R
2. 基础矩阵：K的负转置乘E乘K负逆

求解 F：
$$u_2^TFu_1=0$$
展开：
$$\begin{bmatrix}
   u_x'u_x & u_x'u_y  & u_x' & u_y'u_x & u_y'u_y & u_y' & u_x & u_y & 1 \\
  \end{bmatrix}
  \begin{bmatrix}
   f_1  \\
   f_2  \\
   f_3 \\
   f_4  \\
   f_5 \\
   f_6  \\
   f_7  \\
   f_8  \\
   f_9  \\
  \end{bmatrix}=0
  $$
  $$
  Ax=0
  $$
九个未知数但是只需要八组方程，因为Ax乘任何尺度都为0，所以一般约束
$$
||x||=1，f_9=1
$$

SVD(A),F为最小奇异值的右特征向量，

F矩阵的自由度为7，秩为2，因此需要进行奇异值约束，

SVD(F),

$$
F^*=Udiag(\sigma _1 ,\sigma _2 ,0)V^T
$$

求解E

直接用K矩阵求 

E分解Rt

对E进行SVD分解E应该有两个相等的奇异值

[具体分解过程](https://zhuanlan.zhihu.com/p/434787470)


如何评价 F好不好，

$$
F  \begin{bmatrix}
   u_x  \\
   u_y  \\
   1 \\
  \end{bmatrix}=\begin{bmatrix}
   a  \\
   b \\
   c \\
  \end{bmatrix}
$$

**对极约束与场景无关，在任何场景结构都成立，对极约束能把点约束在一条极线上**

a,b,c是u投影到u’的射线，这是用匹配点到射线的距离就能评价


代码：

**初始化会用F矩阵**

**归一化，选八个点 算F 重投影打分，回复F的尺度，对F进行分解E在分解Rt**


```C++
/****************************************************/
//1.RANSAC流程
void Initializer::FindFundamental(vector<bool> &vbMatchesInliers, float &score, cv::Mat &F21)
{
    // Number of putative matches
    const int N = vbMatchesInliers.size();

    // Normalize coordinates
    vector<cv::Point2f> vPn1, vPn2;
    cv::Mat T1, T2;
    Normalize(mvKeys1,vPn1, T1);
    Normalize(mvKeys2,vPn2, T2);
    cv::Mat T2t = T2.t();

    // Best Results variables
    score = 0.0;
    vbMatchesInliers = vector<bool>(N,false);

    // Iteration variables
    vector<cv::Point2f> vPn1i(8);
    vector<cv::Point2f> vPn2i(8);
    cv::Mat F21i;
    vector<bool> vbCurrentInliers(N,false);
    float currentScore;

    // Perform all RANSAC iterations and save the solution with highest score
    for(int it=0; it<mMaxIterations; it++)
    {
        // Select a minimum set
        for(int j=0; j<8; j++)
        {
            int idx = mvSets[it][j];

            vPn1i[j] = vPn1[mvMatches12[idx].first];
            vPn2i[j] = vPn2[mvMatches12[idx].second];
        }

        cv::Mat Fn = ComputeF21(vPn1i,vPn2i);

        F21i = T2t*Fn*T1;

        // 利用重投影误差为当次RANSAC的结果评分
        currentScore = CheckFundamental(F21i, vbCurrentInliers, mSigma);

        if(currentScore>score)
        {
            F21 = F21i.clone();
            vbMatchesInliers = vbCurrentInliers;
            score = currentScore;
        }
    }
}
/**********************************************************************/
//2.F的计算
//八个点 算A 然后SVD分解A
//最小特征值对应的特征向量。
//整理特征向量成3*3矩阵
//对矩阵SVD分解 进行奇异值约束
cv::Mat Initializer::ComputeF21(const vector<cv::Point2f> &vP1,const vector<cv::Point2f> &vP2)
{
    const int N = vP1.size();

    cv::Mat A(N,9,CV_32F); // N*9

    for(int i=0; i<N; i++)
    {
        const float u1 = vP1[i].x;
        const float v1 = vP1[i].y;
        const float u2 = vP2[i].x;
        const float v2 = vP2[i].y;

        A.at<float>(i,0) = u2*u1;
        A.at<float>(i,1) = u2*v1;
        A.at<float>(i,2) = u2;
        A.at<float>(i,3) = v2*u1;
        A.at<float>(i,4) = v2*v1;
        A.at<float>(i,5) = v2;
        A.at<float>(i,6) = u1;
        A.at<float>(i,7) = v1;
        A.at<float>(i,8) = 1;
    }

    cv::Mat u,w,vt;

    cv::SVDecomp(A,w,u,vt,cv::SVD::MODIFY_A | cv::SVD::FULL_UV);

    cv::Mat Fpre = vt.row(8).reshape(0, 3); // v的最后一列

    cv::SVDecomp(Fpre,w,u,vt,cv::SVD::MODIFY_A | cv::SVD::FULL_UV);

    w.at<float>(2)=0; // 秩2约束，将第3个奇异值设为0

    return  u*cv::Mat::diag(w)*vt;
}

/******************************************************************/
//3.给F矩阵打分 并check 内点
//
float Initializer::CheckFundamental(const cv::Mat &F21, vector<bool> &vbMatchesInliers, float sigma)
{
    const int N = mvMatches12.size();

    const float f11 = F21.at<float>(0,0);
    const float f12 = F21.at<float>(0,1);
    const float f13 = F21.at<float>(0,2);
    const float f21 = F21.at<float>(1,0);
    const float f22 = F21.at<float>(1,1);
    const float f23 = F21.at<float>(1,2);
    const float f31 = F21.at<float>(2,0);
    const float f32 = F21.at<float>(2,1);
    const float f33 = F21.at<float>(2,2);

    vbMatchesInliers.resize(N);

    float score = 0;

    // 基于卡方检验计算出的阈值（假设测量有一个像素的偏差）
    const float th = 3.841;
    const float thScore = 5.991;

    const float invSigmaSquare = 1.0/(sigma*sigma);

    for(int i=0; i<N; i++)
    {
        bool bIn = true;

        const cv::KeyPoint &kp1 = mvKeys1[mvMatches12[i].first];
        const cv::KeyPoint &kp2 = mvKeys2[mvMatches12[i].second];

        const float u1 = kp1.pt.x;
        const float v1 = kp1.pt.y;
        const float u2 = kp2.pt.x;
        const float v2 = kp2.pt.y;

        // Reprojection error in second image
        // l2=F21x1=(a2,b2,c2)
        // F21x1可以算出x1在图像中x2对应的线l
        const float a2 = f11*u1+f12*v1+f13;
        const float b2 = f21*u1+f22*v1+f23;
        const float c2 = f31*u1+f32*v1+f33;

        // x2应该在l这条线上:x2点乘l = 0 
        const float num2 = a2*u2+b2*v2+c2;

        const float squareDist1 = num2*num2/(a2*a2+b2*b2); // 点到线的几何距离 的平方

        const float chiSquare1 = squareDist1*invSigmaSquare;

        if(chiSquare1>th)
            bIn = false;
        else
            score += thScore - chiSquare1;

        // Reprojection error in second image
        // l1 =x2tF21=(a1,b1,c1)

        const float a1 = f11*u2+f21*v2+f31;
        const float b1 = f12*u2+f22*v2+f32;
        const float c1 = f13*u2+f23*v2+f33;

        const float num1 = a1*u1+b1*v1+c1;

        const float squareDist2 = num1*num1/(a1*a1+b1*b1);

        const float chiSquare2 = squareDist2*invSigmaSquare;

        if(chiSquare2>th)
            bIn = false;
        else
            score += thScore - chiSquare2;

        if(bIn)
            vbMatchesInliers[i]=true;
        else
            vbMatchesInliers[i]=false;
    }

    return score;
}

/**************************************************/
//4. 分解F
//先回复E
//分解E 有四种组合  
//对四种组合进行check 并且这里就调用了三角化 把Rt对应的3D点给恢复出来
//先出两个相机视角朝前的组合并且要判定一下视差角足够大
//
bool Initializer::ReconstructF(vector<bool> &vbMatchesInliers, cv::Mat &F21, cv::Mat &K,
                            cv::Mat &R21, cv::Mat &t21, vector<cv::Point3f> &vP3D, vector<bool> &vbTriangulated, float minParallax, int minTriangulated)
{
    int N=0;
    for(size_t i=0, iend = vbMatchesInliers.size() ; i<iend; i++)
        if(vbMatchesInliers[i])
            N++;

    // Compute Essential Matrix from Fundamental Matrix
    cv::Mat E21 = K.t()*F21*K;

    cv::Mat R1, R2, t;

    // Recover the 4 motion hypotheses
    // 虽然这个函数对t有归一化，但并没有决定单目整个SLAM过程的尺度
    // 因为CreateInitialMapMonocular函数对3D点深度会缩放，然后反过来对 t 有改变
    DecomposeE(E21,R1,R2,t);  

    cv::Mat t1=t;
    cv::Mat t2=-t;

    // Reconstruct with the 4 hyphoteses and check
    vector<cv::Point3f> vP3D1, vP3D2, vP3D3, vP3D4;
    vector<bool> vbTriangulated1,vbTriangulated2,vbTriangulated3, vbTriangulated4;
    float parallax1,parallax2, parallax3, parallax4;

    int nGood1 = CheckRT(R1,t1,mvKeys1,mvKeys2,mvMatches12,vbMatchesInliers,K, vP3D1, 4.0*mSigma2, vbTriangulated1, parallax1);
    int nGood2 = CheckRT(R2,t1,mvKeys1,mvKeys2,mvMatches12,vbMatchesInliers,K, vP3D2, 4.0*mSigma2, vbTriangulated2, parallax2);
    int nGood3 = CheckRT(R1,t2,mvKeys1,mvKeys2,mvMatches12,vbMatchesInliers,K, vP3D3, 4.0*mSigma2, vbTriangulated3, parallax3);
    int nGood4 = CheckRT(R2,t2,mvKeys1,mvKeys2,mvMatches12,vbMatchesInliers,K, vP3D4, 4.0*mSigma2, vbTriangulated4, parallax4);

    int maxGood = max(nGood1,max(nGood2,max(nGood3,nGood4)));

    R21 = cv::Mat();
    t21 = cv::Mat();

    // minTriangulated为可以三角化恢复三维点的个数
    int nMinGood = max(static_cast<int>(0.9*N),minTriangulated);

    int nsimilar = 0;
    if(nGood1>0.7*maxGood)
        nsimilar++;
    if(nGood2>0.7*maxGood)
        nsimilar++;
    if(nGood3>0.7*maxGood)
        nsimilar++;
    if(nGood4>0.7*maxGood)
        nsimilar++;

    // If there is not a clear winner or not enough triangulated points reject initialization
    // 四个结果中如果没有明显的最优结果，则返回失败
    if(maxGood<nMinGood || nsimilar>1)
    {
        return false;
    }

    // If best reconstruction has enough parallax initialize
    // 比较大的视差角
    if(maxGood==nGood1)
    {
        if(parallax1>minParallax)
        {
            vP3D = vP3D1;
            vbTriangulated = vbTriangulated1;

            R1.copyTo(R21);
            t1.copyTo(t21);
            return true;
        }
    }else if(maxGood==nGood2)
    {
        if(parallax2>minParallax)
        {
            vP3D = vP3D2;
            vbTriangulated = vbTriangulated2;

            R2.copyTo(R21);
            t1.copyTo(t21);
            return true;
        }
    }else if(maxGood==nGood3)
    {
        if(parallax3>minParallax)
        {
            vP3D = vP3D3;
            vbTriangulated = vbTriangulated3;

            R1.copyTo(R21);
            t2.copyTo(t21);
            return true;
        }
    }else if(maxGood==nGood4)
    {
        if(parallax4>minParallax)
        {
            vP3D = vP3D4;
            vbTriangulated = vbTriangulated4;

            R2.copyTo(R21);
            t2.copyTo(t21);
            return true;
        }
    }

    return false;
}
/***************************************************************/
//分解E为Rt
void Initializer::DecomposeE(const cv::Mat &E, cv::Mat &R1, cv::Mat &R2, cv::Mat &t)
{
    cv::Mat u,w,vt;
    cv::SVD::compute(E,w,u,vt);

    // 对 t 有归一化，但是这个地方并没有决定单目整个SLAM过程的尺度
    // 因为CreateInitialMapMonocular函数对3D点深度会缩放，然后反过来对 t 有改变
    u.col(2).copyTo(t);
    t=t/cv::norm(t);

    cv::Mat W(3,3,CV_32F,cv::Scalar(0));
    W.at<float>(0,1)=-1;
    W.at<float>(1,0)=1;
    W.at<float>(2,2)=1;

    R1 = u*W*vt;
    if(cv::determinant(R1)<0) // 旋转矩阵有行列式为1的约束
        R1=-R1;

    R2 = u*W.t()*vt;
    if(cv::determinant(R2)<0)
        R2=-R2;
}
```

#### （2）单应变换
$$N^TX_1=d$$
相机坐标系下点在点坐在平面法向量上的投影长度为d
$$x_2=K(R+T\frac{1}{d}N^T)K^{-1}x_1$$
1. 加了一个约束：相机坐标系下点在平面法向量上的投影长度为d
1. 单应矩阵：K（R+T(1/d)N置）K的负逆
如何求H：

$$x_2=Hx_1$$

典型的DLT问题 转化成AX=0 然后SVD分解
$$
  \begin{bmatrix}
   0 & 0 & 0 & -x & -y & -1 & xy' & yy' & yy'  \\
   -x & -y & -1 & 0 & 0 & 0 & xx' & yx' & yx' \\
  \end{bmatrix}
    \begin{bmatrix}
    h_1 \\
    h_2 \\
    h_3 \\
    h_4 \\
    h_5 \\
    h_6 \\
    h_7 \\
    h_8 \\
    h_9 \\
    \end{bmatrix}
$$

SVD(A) 最小奇异值对应的有特征向量为H

代码：

**主要用来干什么**

**怎么干的**
```C++
//1.RANSAC
void Initializer::FindHomography(vector<bool> &vbMatchesInliers, float &score, cv::Mat &H21)
{
    // Number of putative matches
    const int N = mvMatches12.size();

    // Normalize coordinates
    // 将mvKeys1和mvKey2归一化到均值为0，一阶绝对矩为1，归一化矩阵分别为T1、T2
    vector<cv::Point2f> vPn1, vPn2;
    cv::Mat T1, T2;
    Normalize(mvKeys1,vPn1, T1);
    Normalize(mvKeys2,vPn2, T2);
    cv::Mat T2inv = T2.inv();

    // Best Results variables
    // 最终最佳的MatchesInliers与得分
    score = 0.0;
    vbMatchesInliers = vector<bool>(N,false);

    // Iteration variables
    vector<cv::Point2f> vPn1i(8);
    vector<cv::Point2f> vPn2i(8);
    cv::Mat H21i, H12i;
    // 每次RANSAC的MatchesInliers与得分
    vector<bool> vbCurrentInliers(N,false);
    float currentScore;

    // Perform all RANSAC iterations and save the solution with highest score
    for(int it=0; it<mMaxIterations; it++)
    {
        // 这个应该最少4对匹配点就可以了
        // Select a minimum set
        for(size_t j=0; j<8; j++)
        {
            int idx = mvSets[it][j];

            // vPn1i和vPn2i为匹配的特征点对的坐标
            vPn1i[j] = vPn1[mvMatches12[idx].first];
            vPn2i[j] = vPn2[mvMatches12[idx].second];
        }

        cv::Mat Hn = ComputeH21(vPn1i,vPn2i);
        // 恢复原始的均值和尺度
        H21i = T2inv*Hn*T1;
        H12i = H21i.inv();

        // 利用重投影误差为当次RANSAC的结果评分
        currentScore = CheckHomography(H21i, H12i, vbCurrentInliers, mSigma);

        // 得到最优的vbMatchesInliers与score
        if(currentScore>score)
        {
            H21 = H21i.clone();
            vbMatchesInliers = vbCurrentInliers;
            score = currentScore;
        }
    }
}
/******************************************************/
// 2.计算H
cv::Mat Initializer::ComputeH21(const vector<cv::Point2f> &vP1, const vector<cv::Point2f> &vP2)
{
    const int N = vP1.size();

    cv::Mat A(2*N,9,CV_32F); // 2N*9

    for(int i=0; i<N; i++)
    {
        const float u1 = vP1[i].x;
        const float v1 = vP1[i].y;
        const float u2 = vP2[i].x;
        const float v2 = vP2[i].y;

        A.at<float>(2*i,0) = 0.0;
        A.at<float>(2*i,1) = 0.0;
        A.at<float>(2*i,2) = 0.0;
        A.at<float>(2*i,3) = -u1;
        A.at<float>(2*i,4) = -v1;
        A.at<float>(2*i,5) = -1;
        A.at<float>(2*i,6) = v2*u1;
        A.at<float>(2*i,7) = v2*v1;
        A.at<float>(2*i,8) = v2;

        A.at<float>(2*i+1,0) = u1;
        A.at<float>(2*i+1,1) = v1;
        A.at<float>(2*i+1,2) = 1;
        A.at<float>(2*i+1,3) = 0.0;
        A.at<float>(2*i+1,4) = 0.0;
        A.at<float>(2*i+1,5) = 0.0;
        A.at<float>(2*i+1,6) = -u2*u1;
        A.at<float>(2*i+1,7) = -u2*v1;
        A.at<float>(2*i+1,8) = -u2;

    }

    cv::Mat u,w,vt;

    cv::SVDecomp(A,w,u,vt,cv::SVD::MODIFY_A | cv::SVD::FULL_UV);

    return vt.row(8).reshape(0, 3); // v的最后一列
}

/******************************************************/
//3.H 算重投影误差
//不用结算Rt直接xH=x‘ 投影过去 算重投影误差

float Initializer::CheckHomography(const cv::Mat &H21, const cv::Mat &H12, vector<bool> &vbMatchesInliers, float sigma)
{   
    const int N = mvMatches12.size();

    // |h11 h12 h13|
    // |h21 h22 h23|
    // |h31 h32 h33|
    const float h11 = H21.at<float>(0,0);
    const float h12 = H21.at<float>(0,1);
    const float h13 = H21.at<float>(0,2);
    const float h21 = H21.at<float>(1,0);
    const float h22 = H21.at<float>(1,1);
    const float h23 = H21.at<float>(1,2);
    const float h31 = H21.at<float>(2,0);
    const float h32 = H21.at<float>(2,1);
    const float h33 = H21.at<float>(2,2);

    // |h11inv h12inv h13inv|
    // |h21inv h22inv h23inv|
    // |h31inv h32inv h33inv|
    const float h11inv = H12.at<float>(0,0);
    const float h12inv = H12.at<float>(0,1);
    const float h13inv = H12.at<float>(0,2);
    const float h21inv = H12.at<float>(1,0);
    const float h22inv = H12.at<float>(1,1);
    const float h23inv = H12.at<float>(1,2);
    const float h31inv = H12.at<float>(2,0);
    const float h32inv = H12.at<float>(2,1);
    const float h33inv = H12.at<float>(2,2);

    vbMatchesInliers.resize(N);

    float score = 0;

    // 基于卡方检验计算出的阈值（假设测量有一个像素的偏差）
    const float th = 5.991;

    //信息矩阵，方差平方的倒数
    const float invSigmaSquare = 1.0/(sigma*sigma);

    // N对特征匹配点
    for(int i=0; i<N; i++)
    {
        bool bIn = true;

        const cv::KeyPoint &kp1 = mvKeys1[mvMatches12[i].first];
        const cv::KeyPoint &kp2 = mvKeys2[mvMatches12[i].second];

        const float u1 = kp1.pt.x;
        const float v1 = kp1.pt.y;
        const float u2 = kp2.pt.x;
        const float v2 = kp2.pt.y;

        // Reprojection error in first image
        // x2in1 = H12*x2
        // 将图像2中的特征点单应到图像1中
        // |u1|   |h11inv h12inv h13inv||u2|
        // |v1| = |h21inv h22inv h23inv||v2|
        // |1 |   |h31inv h32inv h33inv||1 |
        const float w2in1inv = 1.0/(h31inv*u2+h32inv*v2+h33inv);
        const float u2in1 = (h11inv*u2+h12inv*v2+h13inv)*w2in1inv;
        const float v2in1 = (h21inv*u2+h22inv*v2+h23inv)*w2in1inv;

        // 计算重投影误差
        const float squareDist1 = (u1-u2in1)*(u1-u2in1)+(v1-v2in1)*(v1-v2in1);

        // 根据方差归一化误差
        const float chiSquare1 = squareDist1*invSigmaSquare;

        if(chiSquare1>th)
            bIn = false;
        else
            score += th - chiSquare1;

        // Reprojection error in second image
        // x1in2 = H21*x1
        // 将图像1中的特征点单应到图像2中
        const float w1in2inv = 1.0/(h31*u1+h32*v1+h33);
        const float u1in2 = (h11*u1+h12*v1+h13)*w1in2inv;
        const float v1in2 = (h21*u1+h22*v1+h23)*w1in2inv;

        const float squareDist2 = (u2-u1in2)*(u2-u1in2)+(v2-v1in2)*(v2-v1in2);

        const float chiSquare2 = squareDist2*invSigmaSquare;

        if(chiSquare2>th)
            bIn = false;
        else
            score += th - chiSquare2;

        if(bIn)
            vbMatchesInliers[i]=true;
        else
            vbMatchesInliers[i]=false;
    }

    return score;
}

//4. H 回复R t

bool Initializer::ReconstructH(vector<bool> &vbMatchesInliers, cv::Mat &H21, cv::Mat &K,
                      cv::Mat &R21, cv::Mat &t21, vector<cv::Point3f> &vP3D, vector<bool> &vbTriangulated, float minParallax, int minTriangulated)
{
    int N=0;
    for(size_t i=0, iend = vbMatchesInliers.size() ; i<iend; i++)
        if(vbMatchesInliers[i])
            N++;

    // We recover 8 motion hypotheses using the method of Faugeras et al.
    // Motion and structure from motion in a piecewise planar environment.
    // International Journal of Pattern Recognition and Artificial Intelligence, 1988

    // step1：SVD分解Homography
    // 因为特征点是图像坐标系，所以讲H矩阵由相机坐标系换算到图像坐标系
    cv::Mat invK = K.inv();
    cv::Mat A = invK*H21*K;

    cv::Mat U,w,Vt,V;
    cv::SVD::compute(A,w,U,Vt,cv::SVD::FULL_UV);
    V=Vt.t();

    float s = cv::determinant(U)*cv::determinant(Vt);

    float d1 = w.at<float>(0);
    float d2 = w.at<float>(1);
    float d3 = w.at<float>(2);

    // SVD分解的正常情况是特征值降序排列
    if(d1/d2<1.00001 || d2/d3<1.00001)
    {
        return false;
    }

    vector<cv::Mat> vR, vt, vn;
    vR.reserve(8);
    vt.reserve(8);
    vn.reserve(8);

    // step2：计算法向量
    // n'=[x1 0 x3] 4 posibilities e1=e3=1, e1=1 e3=-1, e1=-1 e3=1, e1=e3=-1
    // 法向量n'= [x1 0 x3] 对应ppt的公式17
    float aux1 = sqrt((d1*d1-d2*d2)/(d1*d1-d3*d3));
    float aux3 = sqrt((d2*d2-d3*d3)/(d1*d1-d3*d3));
    float x1[] = {aux1,aux1,-aux1,-aux1};
    float x3[] = {aux3,-aux3,aux3,-aux3};

    // step3：恢复旋转矩阵
    // step3.1：计算 sin(theta)和cos(theta)，case d'=d2
    // 计算ppt中公式19
    float aux_stheta = sqrt((d1*d1-d2*d2)*(d2*d2-d3*d3))/((d1+d3)*d2);

    float ctheta = (d2*d2+d1*d3)/((d1+d3)*d2);
    float stheta[] = {aux_stheta, -aux_stheta, -aux_stheta, aux_stheta};

    // step3.2：计算四种旋转矩阵R，t
    // 计算旋转矩阵 R‘，计算ppt中公式18
    //      | ctheta      0   -aux_stheta|       | aux1|
    // Rp = |    0        1       0      |  tp = |  0  |
    //      | aux_stheta  0    ctheta    |       |-aux3|

    //      | ctheta      0    aux_stheta|       | aux1|
    // Rp = |    0        1       0      |  tp = |  0  |
    //      |-aux_stheta  0    ctheta    |       | aux3|

    //      | ctheta      0    aux_stheta|       |-aux1|
    // Rp = |    0        1       0      |  tp = |  0  |
    //      |-aux_stheta  0    ctheta    |       |-aux3|

    //      | ctheta      0   -aux_stheta|       |-aux1|
    // Rp = |    0        1       0      |  tp = |  0  |
    //      | aux_stheta  0    ctheta    |       | aux3|
    for(int i=0; i<4; i++)
    {
        cv::Mat Rp=cv::Mat::eye(3,3,CV_32F);
        Rp.at<float>(0,0)=ctheta;
        Rp.at<float>(0,2)=-stheta[i];
        Rp.at<float>(2,0)=stheta[i];
        Rp.at<float>(2,2)=ctheta;

        cv::Mat R = s*U*Rp*Vt;
        vR.push_back(R);

        cv::Mat tp(3,1,CV_32F);
        tp.at<float>(0)=x1[i];
        tp.at<float>(1)=0;
        tp.at<float>(2)=-x3[i];
        tp*=d1-d3;

        // 这里虽然对t有归一化，并没有决定单目整个SLAM过程的尺度
        // 因为CreateInitialMapMonocular函数对3D点深度会缩放，然后反过来对 t 有改变
        cv::Mat t = U*tp;
        vt.push_back(t/cv::norm(t));

        cv::Mat np(3,1,CV_32F);
        np.at<float>(0)=x1[i];
        np.at<float>(1)=0;
        np.at<float>(2)=x3[i];

        cv::Mat n = V*np;
        if(n.at<float>(2)<0)
            n=-n;
        vn.push_back(n);
    }

    // step3.3：计算 sin(theta)和cos(theta)，case d'=-d2
    // 计算ppt中公式22
    float aux_sphi = sqrt((d1*d1-d2*d2)*(d2*d2-d3*d3))/((d1-d3)*d2);

    float cphi = (d1*d3-d2*d2)/((d1-d3)*d2);
    float sphi[] = {aux_sphi, -aux_sphi, -aux_sphi, aux_sphi};

    // step3.4：计算四种旋转矩阵R，t
    // 计算旋转矩阵 R‘，计算ppt中公式21
    for(int i=0; i<4; i++)
    {
        cv::Mat Rp=cv::Mat::eye(3,3,CV_32F);
        Rp.at<float>(0,0)=cphi;
        Rp.at<float>(0,2)=sphi[i];
        Rp.at<float>(1,1)=-1;
        Rp.at<float>(2,0)=sphi[i];
        Rp.at<float>(2,2)=-cphi;

        cv::Mat R = s*U*Rp*Vt;
        vR.push_back(R);

        cv::Mat tp(3,1,CV_32F);
        tp.at<float>(0)=x1[i];
        tp.at<float>(1)=0;
        tp.at<float>(2)=x3[i];
        tp*=d1+d3;

        cv::Mat t = U*tp;
        vt.push_back(t/cv::norm(t));

        cv::Mat np(3,1,CV_32F);
        np.at<float>(0)=x1[i];
        np.at<float>(1)=0;
        np.at<float>(2)=x3[i];

        cv::Mat n = V*np;
        if(n.at<float>(2)<0)
            n=-n;
        vn.push_back(n);
    }


    int bestGood = 0;
    int secondBestGood = 0;    
    int bestSolutionIdx = -1;
    float bestParallax = -1;
    vector<cv::Point3f> bestP3D;
    vector<bool> bestTriangulated;

    // Instead of applying the visibility constraints proposed in the Faugeras' paper (which could fail for points seen with low parallax)
    // We reconstruct all hypotheses and check in terms of triangulated points and parallax
    // step4：d'=d2和d'=-d2分别对应8组(R t)，通过恢复3D点并判断是否在相机正前方的方法来确定最优解
    for(size_t i=0; i<8; i++)
    {
        float parallaxi;
        vector<cv::Point3f> vP3Di;
        vector<bool> vbTriangulatedi;
        int nGood = CheckRT(vR[i],vt[i],mvKeys1,mvKeys2,mvMatches12,vbMatchesInliers,K,vP3Di, 4.0*mSigma2, vbTriangulatedi, parallaxi);

        // 保留最优的和次优的
        if(nGood>bestGood)
        {
            secondBestGood = bestGood;
            bestGood = nGood;
            bestSolutionIdx = i;
            bestParallax = parallaxi;
            bestP3D = vP3Di;
            bestTriangulated = vbTriangulatedi;
        }
        else if(nGood>secondBestGood)
        {
            secondBestGood = nGood;
        }
    }

    // step5：通过判断最优是否明显好于次优，从而判断该次Homography分解是否成功
    if(secondBestGood<0.75*bestGood && bestParallax>=minParallax && bestGood>minTriangulated && bestGood>0.9*N)
    {
        vR[bestSolutionIdx].copyTo(R21);
        vt[bestSolutionIdx].copyTo(t21);
        vP3D = bestP3D;
        vbTriangulated = bestTriangulated;

        return true;
    }

    return false;
}
```
