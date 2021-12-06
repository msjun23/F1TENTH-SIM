import cv2

def Img2SimCoordinate(img_coordinate):
    x_sim = img_coordinate[0] * 0.08534 - 156.68159080844768
    y_sim = (2000 - img_coordinate[1]) * 0.08534 - 121.23484964729177
    return [x_sim, y_sim]
        
def Sim2ImgCoordinate(sim_coordinate):
    x_img = (sim_coordinate[0] + 156.68159080844768) / 0.08534
    y_img = 2000 - ((sim_coordinate[1] + 121.23484964729177) / 0.08534)
    return [x_img, y_img]

def onMouse(event, x, y, flags, param):
    if event==cv2.EVENT_LBUTTONDOWN:
        print('왼쪽 마우스 클릭 했을 때 좌표: ', x, y)
        print('Sim 좌표: ', Img2SimCoordinate([x, y]))

img = cv2.imread('pkg/maps/ROBOT_NAVIGATION.png')

cv2.imshow('image', img)
cv2.setMouseCallback('image', onMouse)

cv2.waitKey()
cv2.destroyAllWindows()