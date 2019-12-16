# cap-detect

##### 运行sift.py前需要修改的地方

* 需要检测的图片放在train文件夹下
* 标准图片放在query文件夹下，需要以下面的方式append到对应瓶盖姿态的img列表中

```
# load query images
img = [[], [], []]
img[FRONT].append(cv2.imread('query/front.png',0))
img[FRONT].append(cv2.imread('query/front-1.png', 0))
img[BACK].append(cv2.imread('query/back.png',0))
img[SIDE].append(cv2.imread('query/side.png',0))
img[SIDE].append(cv2.imread('query/side-1.png', 0))
```

* MIN_MATCH_COUNT为匹配数的threshold，设为20以下出来的结果都差不多，可以不用改

