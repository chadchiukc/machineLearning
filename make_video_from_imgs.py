import cv2
import os

fps = 10
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
folders = os.listdir('img')
for folder in folders:
    if folder == '.DS_Store':
        continue
    if 'ladder' in folder:
        video_name = 'video/' + folder + '.mp4'
        videoWrite = cv2.VideoWriter(video_name, fourcc, fps, (1440, 720))
        files = os.listdir('img/'+folder+'/')
        # out_num = len(files)
        # print(folder)
        # for i in range(0, out_num):
        #     fileName = "img/"+folder+'/'+str(i).zfill(4)+'.png'
        #     img = cv2.imread(fileName)
        #     videoWrite.write(img)
        for file in files:
            print(file)
            os.chdir('img/'+folder+'/')
            print(os.path.isfile(file))
            img = cv2.imread(file, cv2.IMREAD_ANYCOLOR)
            print(img)
            videoWrite.write(img)

        cv2.destroyAllWindows()
        videoWrite.release()

