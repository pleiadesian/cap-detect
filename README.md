# cap-detect

* 标准图片放在query目录下对应名字的目录下

```
query
├── back
│   ├── 瓶口朝下的瓶盖图片.jpg/.png
│   └── 瓶口朝下的瓶盖图片经过标注的数据.json
├── front
│   ├── 瓶口朝上的瓶盖图片.jpg/.png
│   └── 瓶口朝上的瓶盖图片经过标注的数据.json
└── side
    ├── 侧着的瓶盖图片.jpg/.png
    └── 侧着的瓶盖图片经过标注的数据.json
```

* 标注数据
  * 与标准图像同名
  * 瓶盖边缘标签名标为“edge”
  * 瓶盖原点标签名标为“origin”
* HOG descriptor
  * 如果hog目录下已经有算好的数据，sift.py的第75行直接读取hog目录下的数据比较快
  * 如果hog目录下没有算好的数据，或者标准集有更新过，请手动修改下面的注释，使得fd通过调用hog_des计算得到

```
# Initiate HOG fd
# fd = hog.hog_des(img_hog)
fd = hog.load_fd()
```

* 调用方法如下，sift_init只需调用一次，之后每一次用调用sift_match匹配图片时需要将sift_init返回的东西传进去

```
query_img0, query_img_hog0, query_img_name0, kp0, des0 = sift_init()

input_image0 = cv2.imread("train/test.png")
selected0, img_mask0, origin_point0 = sift_match(input_image0, query_img0, query_img_hog0, query_img_name0, kp0, des0)
```

