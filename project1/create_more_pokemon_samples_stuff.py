import pandas as pd
from pathlib import Path
import cv2

'''
pokemon = pd.read_csv('C:/Users/GonVirginia/Desktop/AP Aulas/More Pokemon/csv/Pokemon.csv')
print(pokemon)
pokemon = pokemon[['#', 'Type 1']]
print(pokemon)
pokemon.to_csv('C:/Users/GonVirginia/Desktop/AP Aulas/More Pokemon/csv/AllPokemonType.csv', index=False)

pokemon = pd.read_csv('C:/Users/GonVirginia/Desktop/AP Aulas/More Pokemon/csv/AllPokemonType.csv')
pokemon = pokemon.rename(columns={'#':'image_id', 'Type 1':'main_type'})
pokemon.to_csv('C:/Users/GonVirginia/Desktop/AP Aulas/More Pokemon/csv/AllPokemonTypeFormatted.csv', index=False)

pokemon = pd.read_csv('C:/Users/GonVirginia/Desktop/AP Aulas/More Pokemon/csv/AllPokemonTypeFormatted.csv')
pokemon['image_id'] = 'gen5-' + pokemon['image_id'].astype(str)
pokemon.to_csv('C:/Users/GonVirginia/Desktop/AP Aulas/More Pokemon/csv/AllPokemonTypeFormattedGen5.csv', index=False)

root_dir = Path('C:/Users/GonVirginia/Desktop/AP Aulas/More Pokemon/gen5')
file_paths = root_dir.iterdir()

for path in file_paths:
    new_filename = 'gen5-' + path.stem + path.suffix
    new_filepath = Path(new_filename)
    print(new_filepath)
    path.rename('C:/Users/GonVirginia/Desktop/AP Aulas/More Pokemon/gen5/' + str(new_filepath))

import urllib.request
from time import sleep

for i in range(1000, 2000):
    urllib.request.urlretrieve("https://picsum.photos/100", f"C:/Users/GonVirginia/Desktop/random backgrounds/randombg{i}.jpg")
    sleep(0.1)
'''

from PIL import Image
import random

root_dir = Path('C:/Users/GonVirginia/Desktop/AP Aulas/More Pokemon/all gens')
file_paths = root_dir.iterdir()

for path in file_paths:
    i = random.randint(0, 999)
    background = Image.open(f'C:/Users/GonVirginia/Desktop/AP Aulas/More Pokemon/random backgrounds/randombg{i}.jpg')
    background = background.convert('RGBA')
    pokemon = Image.open(str(path))
    pokemon = pokemon.convert('RGBA')

    bg_w, bg_h = background.size
    pokemon_w, pokemon_h = pokemon.size
    offset = ((bg_w - pokemon_w) // 2, (bg_h - pokemon_h) // 2)

    background.paste(im=pokemon, box=offset, mask=pokemon)
    background = background.convert('RGB')
    background.save(f'C:/Users/GonVirginia/Desktop/AP Aulas/More Pokemon/all gens bg/{path.name}')