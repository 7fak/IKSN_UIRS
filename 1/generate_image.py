from PIL import Image, ImageDraw, ImageFont

def text(symbol, output_path, font="arial.ttf", size=64):
    image = Image.new("RGB", (size, size), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("arial.ttf", size-10)
    draw.text((7, 5), symbol, font=font, fill=(0,0,0))
    image.save(output_path)

if __name__ == "__main__":
    symbols = [str(i) for i in range(10) ]
    for s in symbols:
        text(s, '{}.png'.format(s))