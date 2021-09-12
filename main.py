import paddlehub as hub
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# 输出文件夹
out_path = 'result'
# 待预测图片
img_paths = ["test.jpg", 'test2.jpg']
module = hub.Module(name="pyramidbox_lite_server_mask")
# 可以多张图片的填入列表
imgs = []
for img_path in img_paths:
    result_list = module.face_detection(images=[cv2.imread(img_path)],
                                        use_multi_scale=True,
                                        shrink=0.6,
                                        visualization=True,
                                        output_dir=out_path)
    result = result_list[0]
    for i in result['data']:
        print(i)
    # 除去path最后的".0"
    path_charlist = list(result["path"])
    path_charlist = path_charlist[:-2]
    path_str = ''.join(path_charlist)
    path = out_path + '/' + path_str + '.jpg'
    # print(path)
    img = mpimg.imread(path)
    plt.figure(figsize=(10, 10))
    plt.imshow(img)
    plt.show()


