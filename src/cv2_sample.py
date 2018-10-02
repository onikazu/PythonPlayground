# import cv2
#
# # making instance
# # we can choose camera by selecting argument
# cap = cv2.VideoCapture(0)
#
# while True:
#     print("Start taking video!")
#     # ret is boolean, catch the flag whether we can get frame
#     ret, frame = cap.read()
#
#     cv2.imshow('Raw Frame', frame)
#
#     # esc key makes this loop break
#     k = cv2.waitKey(1)
#     if k == 27:
#         break
#
# cap.release()
# cv2.destroyAllWindows()
# print("release the camera!")

# OpenCV のインポート
import cv2


def concat_tile(im_list_2d):
    return cv2.vconcat([cv2.hconcat(im_list_h) for im_list_h in im_list_2d])


# VideoCaptureのインスタンスを作成する。
# 引数でカメラを選べれる。
cap = cv2.VideoCapture(0)

origin_images = []
for i in range(1, 9):
    origin_images.append(cv2.imread('../image/00000{0:d}.jpg'.format(i)))

frame_num = 0
while True:
    # VideoCaptureから1フレーム読み込む
    ret, frame = cap.read()

    print(len(frame), len(origin_images[0]))
    # 画像変数初期化
    images = []

    # 撮影ビデオ大きさ調整
    frame = cv2.resize(frame, (178, 218))

    # num = 0~7
    num = frame_num % 8

    for i in range(num, num + 8):
        if i > 7:
            images.append(origin_images[i % 8])
        else:
            images.append(origin_images[i])

    # 何か処理（ここでは文字列「hogehoge」を表示する）
    edframe = frame
    cv2.putText(edframe, 'hogehoge', (0, 50), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3, cv2.LINE_AA)

    # 加工済の画像を表示する
    im_tile = concat_tile([[images[0], images[1], images[2]],
                           [images[7], frame, images[3]],
                           [images[6], images[5], images[4]]])

    cv2.imshow('tile camera', im_tile)
    frame_num += 1

    k = cv2.waitKey(1)
    if k == 27:
        print("released!")
        break

# キャプチャをリリースして、ウィンドウをすべて閉じる
cap.release()
cv2.destroyAllWindows()
print("release camera!!!")
