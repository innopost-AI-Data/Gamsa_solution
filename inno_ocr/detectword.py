import cv2
import glob
import os
import itertools
import numpy as np
from tqdm import tqdm

def convertCoordination():
    my_loc = []
    len_loc = []
    txt_path = r'./inno_ocr/img/'
    txt_dir = glob.glob(txt_path + '/*.txt')
    img_path = r'./inno_ocr/img/'
    img_dir = glob.glob(img_path + '/*.jpg')
    for _img, _txt in tqdm(zip(img_dir, txt_dir)):
        # 빈 리스트 생성
        coordinate = []
        # dh dw : 이미지의 가로, 세로 크기
        img_array = np.fromfile(_img, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        # img = cv2.imread(_img)
        dh, dw, _ = img.shape
        # RBOX 좌표가 있는 텍스트 파일 불러옴

        fl = open(_txt, 'r')
        data = [line[:-1] for line in fl.readlines()]
        fl.close()

        if len(data) == 0:
            continue
        for dt in data:
            # Split string to float
            # # l : left , r : right , t : top, b : bottom
            # # x = l , y = t , w = r-l, h = b-t

            x1, y1, x2, y2, x3, y3, x4, y4 = map(int,map(float, dt.split(',')))
            x = x1
            y = y1
            w = x2 - x1
            h = y3 - y2

            # 리스트에 저장
            coordinate.append([x, y, w, h])

        coordinate.sort(key=lambda x: x[1])

        s = []
        loc = []

        for i in range(1, len(coordinate)):
            if abs(coordinate[i - 1][1] - coordinate[i][1]) < 10:
                s.append(coordinate[i - 1])
            else:
                if len(s) == 0:
                    loc.append([coordinate[i - 1]])
                else:
                    s.append(coordinate[i - 1])
                    s.sort(key=lambda x: x[0])
                    loc.append(s)
                    s = []
        if len(coordinate) == 1:
            s.append(coordinate[0])
        else:
            s.append(coordinate[i])
        s.sort(key=lambda x: x[0])
        loc.append(s)

        len_loc.append(list(map(lambda x: len(x), loc)))
        my_loc.append(list(itertools.chain(*loc)))
    return len_loc, my_loc

#만들어낸 loc 리스트를 불러와 좌표값에 따른 이미지 추출하여 저장
def read_img_by_coord (loc):
    img_list=[]
    img_path = r'./inno_ocr/img/'
    img_dir = glob.glob(img_path + '/*.jpg')
    print(len(img_dir))
    for _img,_loc in zip(img_dir, loc):
        print("make image", os.path.basename(_img[:-4]))
        img_array = np.fromfile(_img, np.uint8)
        org_image = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
        # org_image = cv2.imread(_img,cv2.IMREAD_GRAYSCALE)
        cv_list = []
        for i in _loc:
            # y:y+h, x:x+w
            img_trim = org_image[i[1]:i[1]+i[3], i[0]:i[0]+i[2]] #trim한 결과를 img_trim에 담는다
            #dst = cv2.bitwise_not(img_trim)
            cv_list.append(img_trim)
        img_list.append(cv_list)
    return img_list

