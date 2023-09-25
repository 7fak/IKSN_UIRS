from PIL import Image, ImageDraw, ImageFont
import random

def text(symbol, output_path, size=100, RI_size=32):
    image = Image.new("RGB", (size, size), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("./fonts/CoopBlack_Cyrillic_0.ttf", RI_size-5)
    draw.text((random.randint(0, size-RI_size), random.randint(0, size-RI_size)), symbol, font=font, fill=(0, 0, 0))
    image.save(output_path)

if __name__ == "__main__":
    symbols = [str(i) for i in range(10) ]
    for s in symbols:
        text(s, './img/CI_{}.png'.format(s),RI_size=32)