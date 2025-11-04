from pathlib import Path
import requests
import json


path = Path("portraits")
path.mkdir(exist_ok=True)
out_path = Path("segmented")
out_path.mkdir(exist_ok=True)


def download():
    from pywikibot import FilePage, Site
    from pywikibot.pagegenerators import GeneratorFactory
    
    def download_file(url):
        from urllib.parse import unquote
        filename = unquote(url.split('/')[-1])
        filename = filename.replace("01.", "00.")
        local_filename = path.joinpath(filename)
        if local_filename.exists():
            return
        with requests.get(url, stream=True, headers={'user-agent': 'Bot by User:PetraMagna'}) as r:
            r.raise_for_status()
            with open(local_filename, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192): 
                    f.write(chunk)
        print(f"{local_filename.name} downloaded")
        return local_filename
    
    s = Site()
    gen = GeneratorFactory(s)
    gen.handle_args(['-category:Character sprites', '-ns:File', r'-titleregex:0[01]\.'])
    gen = gen.getCombinedGenerator(preload=True)
    for file in gen:
        file: FilePage
        url = file.get_file_url()
        download_file(url)

result_file = "seg-result.json"
manual_file = "seg-result-manual.json"

def predict():
    from groundingdino.util.inference import load_model, load_image, predict, annotate
    import cv2
    model = load_model("groundingdino/config/GroundingDINO_SwinT_OGC.py", "weights/groundingdino_swint_ogc.pth")
    model = model.to('cuda:0')
    TEXT_PROMPT = "face"
    BOX_TRESHOLD = 0.25
    TEXT_TRESHOLD = 0.20
    
    result: dict[str, list[float]] = {}

    for image_path in path.glob("*.png"):
        image_source, image = load_image(image_path)
        boxes, logits, phrases = predict(
            model=model,
            image=image,
            caption=TEXT_PROMPT,
            box_threshold=BOX_TRESHOLD,
            text_threshold=TEXT_TRESHOLD
        )
        if len(boxes) == 0:
            print(f"Warning: cannot segment {image_path.name}")
            continue
        # Uncomment this when debugging: you can see the resulting image
        annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
        cv2.imwrite(str(out_path.joinpath(image_path.name)), annotated_frame)
        center_x, center_y, _, _ = boxes[0]
        center_x, center_y = center_x.item(), center_y.item()
        height, width, _ = image_source.shape
        file_name = image_path.name.replace("_00.png", "")
        result[file_name] =[width, height, center_x, center_y]
    
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4, ensure_ascii=False)

def process():
    result: dict = json.load(open(result_file, "r", encoding="utf-8"))
    for i in range(0, 4):
        print("{{#switch:{{{2|}}}")
        for k, v in result.items():
            print(f"|{k}={v[i]}")
        print("}}")
        
def make_stylesheet():
    result: dict = json.load(open(result_file, "r", encoding="utf-8"))
    multiplier = 4.2
    for k in sorted(result):
        k: str
        v = result[k]
        width, height, x, y = v
        style = f".story-image-{k.replace('_00.png', '').replace('(', '_').replace(')', '_').replace(',', '_')} img {{ " \
                f"margin-left: -{round(max(0, width / multiplier * x - 29), 2)}px; " \
                f"margin-top: -{round(max(0, height / multiplier * y - 29), 2)}px " \
                "}"
        print(style)

def make_height_switch():
    result: dict = json.load(open(result_file, "r", encoding="utf-8"))
    for k, v in result.items():
        width, height, x, y = v
        if height == 1280:
            continue
        k: str
        k = k.replace('_', ' ')
        print(f"|{k}={height}")

def process3():
    result: dict = json.load(open(result_file, "r", encoding="utf-8"))
    manual: dict = json.load(open(manual_file, "r", encoding="utf-8"))
    out_path.mkdir(exist_ok=True)
    for k, v in manual.items():
        result[k] = v
    portrait_width, portrait_height = 300, 300
    for k, v in result.items():
        width, height, center_x, center_y = v
        center_x, center_y = width * center_x, height * center_y
        top_left = [int(round(x)) for x in (center_x - portrait_width / 2, center_y - portrait_height / 2)]
        top_left = [f"+{x}" if x >= 0 else f"{x}" for x in top_left]
        print(f'mogrify -crop {portrait_width}x{portrait_height}{top_left[0]}{top_left[1]} "{k}_??.png"')

def main():
    from sys import argv
    dispatcher = {
        'download': download,
        'predict': predict,
        'css': make_stylesheet,
        'height': make_height_switch,
    }
    if len(argv) <= 1:
        print("Need at least 1 arg among " + ", ".join(dispatcher.keys()))
        return
    dispatcher[argv[1]]()
    
if __name__ == "__main__":
    main()
