from texture_synthesis import texture_synt,load_sample_image

def synthesize():
    images = ['D20.gif',"D26.gif","D88.gif","D4.gif"]
    for img_path in images:
        img = load_sample_image(img_path)
        texture_synt(img,(600,600),"./textures/"+img_path[:-4]+"/")

if __name__ == "__main__":
    synthesize()
