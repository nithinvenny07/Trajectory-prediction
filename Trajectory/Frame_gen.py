import cv2
def frame_gen(file_path):
    vidObj = cv2.VideoCapture(file_path)
    count = 1
    success = 1
    print("Generating Frames from input video")
    while success:
        success , image = vidObj.read()
        if success == 1:
            cv2.imwrite('C:\\Users\\venny\\Desktop\\new data\\c_3 frames\\'+str(count)+'.jpg', image)
        count+=1
    print("Frames generated")
frame_gen("C:\\Users\\venny\\Downloads\\MoreTvideos\\Copy of 3.mp4")