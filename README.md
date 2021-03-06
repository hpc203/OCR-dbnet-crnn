# OCR-dbnet-crnn
faster-rcnn检测图片里的证件照，dbnet检测证件照里的文字，crnn识别文字。
本套程序旨在实现一个身份证全卡面文字解析的功能，程序里使用到的模型文件从百度网盘下载，
链接: https://pan.baidu.com/s/1-RV0QFbV_0TiuDYJZ5nE5g  密码: 9f9o

在本套程序里，含有3个模块：

(1). faster-rcnn检测图片中的证件照，画出检测矩形框，这个模块的程序依赖opencv

(2). DBNet检测证件照里的文字，这个模块的程序依赖onnxruntime，起初我尝试过使用opencv，
但是加载onnx文件出错，于是只能使用onnxruntime库

(3). crnn识别证件照里的文字，这个模块的程序依赖onnxruntime，起初我尝试过使用opencv，
但是加载onnx文件出错，于是只能使用onnxruntime库


其中第1个模块不是必须的，如果图片里的身份证几乎占满整个图片像素区域，那么这时候可以直接使用dbnet检测证件照里的文字的。
如果图片里有多个证件照，那么这时候可以先使用第1个模块，检测图片里的身份证，然后抠出身份证所在的像素ROI区域，
然后把ROI区域输入到DBNet检测证件照里的文字。
如果图片里的除身份证之外的背景也含有文字，那么这时候需要使用第1个模块，检测定位到图片里的身份证，然后抠出身份证所在的像素ROI区域，
排除无关背景里的文字干扰，然后把ROI区域输入到DBNet检测证件照里的文字。

由于身份证属于私密信息，在这里就不上传测试图片了，你可以拿手机自拍身份证，然后把图片输入到程序里做身份证全卡面文字解析。

不过在本套程序里，在第一个模块：检测图片里的身份证，还有需要改进的地方。假如图片里的身份证是倾斜放置的，这时候faster-rcnn有可能检测不到倾斜的
身份证，又或者检测到倾斜的身份证，但这时的检测框是水平矩形框，包含了无关背景，加入背景里含有文字，那就会造成干扰。
因此在检测图片里的身份证的这个模块里，一种优秀的方案是，在检测到身份证后输出水平矩形框和证件照的4个角点，
我在之前发布的车牌检测程序里就是这么做的。又或者使用关键点检测网络，只检测输出证件照的4个角点，不过这种方案也不是最优的。
假如图片里是手持证件照，那么就可能某些角点被遮挡，这时候检测4个角点的方案就会出现失误。如果是检测输出水平矩形框和证件照的4个角点，那么这时候
使用水平矩形框还能抠出图片里的证件照，而DBNet能检测任意方向的文字的，这时候即使是倾斜证件照输入到DBNet，也不会对文字检测造成很大的干扰。
因此，在检测图片里的身份证的这个模块里，检测输出水平矩形框和证件照的4个角点是最优方案
