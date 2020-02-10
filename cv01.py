import cv2

img=cv2.imread("images/thomas.jpg")

cv2.imshow("show the images",img)
cv2.waitKey(0)
cv2.destroyAllWindows()
