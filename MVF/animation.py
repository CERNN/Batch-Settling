from PIL import Image
import glob
 
# def createAnimation():
#     # Create the frames
#     frames = []
#     imgs = glob.glob("./MVF/temporaryFiles/animationFrames/*.png")
#     for i in imgs:
#         new_frame = Image.open(i)
#         frames.append(new_frame)
#     # Save into a GIF file that loops forever
#     frames[0].save('png_to_gif.gif', format='GIF',
#                 append_images=frames[1:],
#                 save_all=True,
#                 duration=20, loop=0)

def createAnimation():
    # Create the frames
    frames = []
    
    for i in range(0,365):
        imgs = glob.glob("./MVF/temporaryFiles/animationFrames/Concentration" + str(i) + ".png")
        for img in imgs:
            new_frame = Image.open(img)
            frames.append(new_frame)
    # Save into a GIF file that loops forever
    frames[0].save('./MVF/temporaryFiles/animationFrames/animation.gif', format='GIF',
                append_images=frames[1:435],
                save_all=True,
                duration=50, loop=1000)

createAnimation()