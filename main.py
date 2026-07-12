from numpy import ndarray


from __future__ import annotations
from tkinter import N
import cv2
from math import inf
from pathlib import Path
from PIL import Image
import numpy as np
from typing import Any, Final



def list_images(directory: Path) -> list[Path]:
    extensions = {".png", ".jpg", ".jpeg"}
    return sorted(
        p for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in extensions
    )

def load_rgba(path: Path) -> np.ndarray:
    img = Image.open(path).convert("RGBA")
    return np.array(img)  # zawsze (H, W, 4)

def get_img_map(paths : list[Path]) -> dict[Path, np.ndarray]:
    ret = {}
    for path in paths:
        try:
            img = Image.open(path)
        except IOError:
            continue
        pixels = np.array(img)
        ret[path] = pixels

    return ret

class MatchOutputInfo:
    def __init__(self, score: float, x_shift: int, y_shift: int, height: int):
        self.score = score
        self.x_shift = x_shift
        self.y_shift = y_shift
        self.height = height

    def get_values_from(self, other: MatchOutputInfo):
        self.score = other.score
        self.x_shift = other.x_shift
        self.y_shift = other.y_shift
        self.height = other.height

def single_match_similarity_score(first: np.ndarray, second: np.ndarray, patter_height: int, acceptable_margines: int) -> MatchOutputInfo:
    # Wycinamy wzorzec z SAMEJ GÓRY (Y: od 0 do 150), ale UCINAMY BOKI (X: od 60 do -60)
    wzorzec = second[0:patter_height, acceptable_margines:-acceptable_margines]

    try:
        # 3. Magia OpenCV - przeszukujemy obraz w osiach X i Y naraz!
        # TM_SQDIFF_NORMED szuka miejsca, gdzie różnica między pikselami dąży do zera.
        wynik_dopasowania = cv2.matchTemplate(first, wzorzec, cv2.TM_SQDIFF_NORMED)

        # Funkcja minMaxLoc znajduje dla nas skrajne wartości w macierzy wyników.
        # Przy metodzie SQDIFF (Square Difference), idealne dopasowanie to najmniejsza wartość.
        min_wartosc, max_wartosc, min_kordy, max_kordy = cv2.minMaxLoc(wynik_dopasowania)

        # 4. Obliczamy faktyczne przesunięcie
        # min_kordy to krotka (X, Y) mówiąca, gdzie w img1 znaleziono nasz wzorzec.
        znalezione_x, znalezione_y = min_kordy

        # UWAGA: Ponieważ ucięliśmy 60 pikseli z lewej strony wzorca,
        # musimy to uwzględnić, aby policzyć o ile przesunął się cały obraz!
        przesuniecie_x = znalezione_x - acceptable_margines
        przesuniecie_y = znalezione_y

        # print(f"Drugi screen nakłada się na pierwszy w punkcie: Y={przesuniecie_y}, X={przesuniecie_x}   dopasowanie {min_wartosc}")
        print(f"score: {min_wartosc}")
        print(f"y_shift: {przesuniecie_y}")
        print(f"x_shift: {przesuniecie_x}")
        print(f"patter_height: {patter_height}")
        return MatchOutputInfo(min_wartosc, przesuniecie_y, przesuniecie_x, patter_height)

    except cv2.error as e:
        return MatchOutputInfo(inf, 0, 0, 0)

def look_for_first_match(inputs: list[Path, np.ndarray]) -> bool:
    acceptable_margines = 60

    for first_idx in range(inputs):
        for second_idx in range(inputs):   # ogarniamy przykłady w obydwie strony, wtedy możemy pozostać ze sprawdzaniemy tylko od góry

            if first_idx == second_idx:
                continue

            first_path, first_pixels = our_dict[first_idx]
            second_path, second_pixels = our_dict[second_idx]

            print(f"first.path: {first_path} first.pixels.shape: {first_pixels.shape}")
            print(f"second.path: {second_path} second.pixels.shape: {second_pixels.shape}\n")

            patter_height = 10

            # first iteration -> starting point #
            current_MatchOutputInfo = single_match_similarity_score(first_pixels, second_pixels, patter_height, acceptable_margines)

            while True:
                patter_height += 1
                matchOutputInfo = single_match_similarity_score(first_pixels, second_pixels, patter_height, acceptable_margines)

                if(0.001 < matchOutputInfo.score):
                    break

                if(matchOutputInfo.score <= current_MatchOutputInfo.score):   # może też zakomentować, ale zobaczymy
                    current_MatchOutputInfo.get_values_from(matchOutputInfo)

            print(f"Biggest pattern height {current_MatchOutputInfo.height}")

            if(0.001 < current_MatchOutputInfo.score):
                continue

            # mergujemy obrazy i zaczynamy całe algo jeszcze raz #




            # Generalny Opis #
            # marginesy po obu stronach na co najmniej wartość różnicy szerokości pomiędzy dwoma obrazami z każdej strony
            # z dołu co najmniej na wysokość jednego + drugiego
            # jako tło używamy koloru, który nie występuje w żadnym z obrazów, z kanałem alpha to możliwe
            # wklejamy obraz second w obraz first -> zapisujemy jako nowy obraz i kasujemy poprzednie, first i second (nowy zapisujemy jako merged_{nazwa_first}_{nazwa_second})
            # potem odcinamy te krawędzie, co sami zrobiliśmy (nieobecny na obrazach kolor)

            #  startujemy całe algo mergujące jeszcze raz, żeby teraz uwzględniać nowy zmergowany plik


            return True

    return False




def main():
    in_dir = Path("in")
    in_dir.mkdir(parents=True, exist_ok=True)

    out_dir = Path("out")
    out_dir.mkdir(parents=True, exist_ok=True)
    img_maps = get_img_map(list_images(in_dir))

    our_dict = list[tuple[Path, ndarray]](img_maps.items())[:3]
    our_dict_length = len(our_dict)




    while look_for_first_match(our_dict):
        pass











if __name__ == "__main__":
    main()
