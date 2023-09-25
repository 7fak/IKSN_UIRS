from PIL import Image, ImageDraw, ImageFont
import random

def text(symbol, output_path, font="arial.ttf", size=100, RI_size=32):
    image = Image.new("RGB", (size, size), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("arial.ttf", RI_size-2)
    draw.text((random.randint(0, size-64), random.randint(0, size-64)), symbol, font=font, fill=(0, 0, 0))
    image.save(output_path)

if __name__ == "__main__":
    symbols = [str(i) for i in range(10) ]
    for s in symbols:
        text(s, './img/CI_{}.png'.format(s),RI_size=32)