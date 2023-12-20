from PIL import Image, ImageDraw
import torchvision.transforms as transforms
transform = transforms.ToTensor()
def get_skeleton_img(size:tuple, pred:list, conf:float, width):
    def toTuple(np_array):
        return tuple([tuple(e) for e in np_array])
    
    def draw_head(d_img, result, conf, width):
        if result[4][2] > conf:
            point = [result[6][:-1].tolist(), result[4][:-1].tolist()]
            d_img.line(toTuple(point), width = width)
        if result[2][2] > conf:
            point = [result[4][:-1].tolist(), result[2][:-1].tolist()]
            d_img.line(toTuple(point), width = width)
        if result[0][2] > conf:
            point = [result[2][:-1].tolist(), result[0][:-1].tolist()]
            d_img.line(toTuple(point), width = width)
        if result[1][2] > conf:
            point = [result[0][:-1].tolist(), result[1][:-1].tolist()]
            d_img.line(toTuple(point), width = width)
        if result[3][2] > conf:
            point = [result[1][:-1].tolist(), result[3][:-1].tolist()]
            d_img.line(toTuple(point), width = width)
        if result[5][2] > conf:
            point = [result[3][:-1].tolist(), result[5][:-1].tolist()]
            d_img.line(toTuple(point), width = width)

    def draw_right_hand(d_img, result, conf, width):
        if result[8][2] > conf:
            point = [result[6][:-1].tolist(), result[8][:-1].tolist()]
            d_img.line(toTuple(point), width = width)
        if result[10][2] > conf:
            point = [result[8][:-1].tolist(), result[10][:-1].tolist()]
            d_img.line(toTuple(point), width = width)

    def draw_left_hand(d_img, result, conf, width):
        if result[7][2] > conf:
            point = [result[5][:-1].tolist(), result[7][:-1].tolist()]
            d_img.line(toTuple(point), width = width) 
        if result[9][2] > conf:
            point = [result[7][:-1].tolist(), result[9][:-1].tolist()]
            d_img.line(toTuple(point), width = width)

    def draw_body(d_img, result, conf, width):
        if result[5][2] > conf:
            point = [result[6][:-1].tolist(), result[5][:-1].tolist()]
            d_img.line(toTuple(point), width = width)
        if result[11][2] > conf:
            point = [result[5][:-1].tolist(), result[11][:-1].tolist()]
            d_img.line(toTuple(point), width = width)
        if result[12][2] > conf:
            point = [result[11][:-1].tolist(), result[12][:-1].tolist()]
            d_img.line(toTuple(point), width = width)
        if result[6][2] > conf:
            point = [result[12][:-1].tolist(), result[6][:-1].tolist()]
            d_img.line(toTuple(point), width = width)

    def draw_right_leg(d_img, result, conf, width):
        if result[14][2] > conf:
            point = [result[12][:-1].tolist(), result[14][:-1].tolist()]
            d_img.line(toTuple(point), width = width) 
        if result[14][2] > conf:
            point = [result[14][:-1].tolist(), result[16][:-1].tolist()]
            d_img.line(toTuple(point), width = width)

    def draw_left_leg(d_img, result, conf, width):
        if result[13][2] > conf:
            point = [result[11][:-1].tolist(), result[13][:-1].tolist()]
            d_img.line(toTuple(point), width = width)
        if result[15][2] > conf:
            point = [result[13][:-1].tolist(), result[15][:-1].tolist()]
            d_img.line(toTuple(point), width = width)
            
    def draw_img(size:tuple, pred:list, conf:float, width):
        img = Image.new("RGB", size)  
        d_img = ImageDraw.Draw(img)
        for result in pred:
            result = result[7:].reshape(17, -1)
            draw_head(d_img, result, conf, width)
            draw_right_hand(d_img, result, conf, width)
            draw_left_hand(d_img, result, conf, width)
            draw_body(d_img, result, conf, width)
            draw_right_leg(d_img, result, conf, width)
            draw_left_leg(d_img, result, conf, width)
        # return transform(img).unsqueeze(0)
        return img

    return draw_img(size, pred, conf, width)