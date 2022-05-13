## 如何用git在github上管理工程
1. [gitbash下载地址](https://git-scm.com/downloads)
2. 步骤
##### 下载 github上的工程
```shell
git init
git pull https://github.com/LeiXu1999/SLAM-Learning-Notes.git
```
##### 上传工程至 github
```shell
git init
git status
git add 想要添加的文件
git add -A
git commit -m " first commit"
git remote add origin https://github.com/LeiXu1999/New Repository
git push origin master
#第一次需要登录 用户名+密钥
```
##### 更新已经上传文件
```shell
git add 想要添加的文件
git add -A
git commit -m " second commit"
# git remote add origin https://github.com/LeiXu1999/New Repository
git push origin master
```
