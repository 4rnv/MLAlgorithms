import pandas as pd
import random

def get_color_label(r, g, b):
    if r > g and r > b:
        return "red"
    elif g > r and g > b:
        return "green"
    elif b > r and b > g:
        return "blue"
    elif r == g and r == b:
        return "gray"
    elif r == g:
        return "yellow" if r > b else "cyan"
    elif g == b:
        return "magenta" if g > r else "cyan"
    elif r == b:
        return "magenta" if r > g else "yellow"
    else:
        return "black"

def create(n,filename):
    data = []
    for _ in range(n):
        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)
        label = get_color_label(r, g, b)
        data.append([r, g, b, label])
    df = pd.DataFrame(data, columns=["R", "G", "B", "Label"])
    df.to_csv(filename, index=False)
    print(f"File {filename} generated with {n} rows.")

create(1000,"train.csv")
create(200,"test.csv")