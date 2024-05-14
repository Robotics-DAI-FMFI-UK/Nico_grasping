import matplotlib.pyplot as plt
import glob

def vytvor_histogram(ax, hodnota1, hodnota2, hodnota3, hodnota4=None):
    # Definícia hodnôt a ich početností
    hodnoty = ['Hodnota 1', 'Hodnota 2', 'Hodnota 3']
    vysky = [hodnota1, hodnota2, hodnota3]
    if hodnota4:
        vysky.append(hodnota4)
        hodnoty.append('Hodnota 4')

    farby = ['red', 'green', 'blue', 'yellow']

    # Vytvorenie histogramu
    ax.bar(hodnoty, vysky, color = farby)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

# Nájdenie všetkých súborov s príponou .vis, .mot, .act
subory = glob.glob("*.vis") + glob.glob("*.mot") + glob.glob("*.act")

# Pre každý súbor
for num,subor in enumerate(subory):
    try:
        hodnoty = []
        with open(subor, 'r') as file:
            loaded_file = file.read().split('\n')
            rozmer = int(loaded_file[0].split(',')[0])
            fig, axs = plt.subplots(rozmer, rozmer, figsize=(7, 7))
            for line in loaded_file[1:]:
                hodnoty.append(tuple(float(i) for i in line.split(',')))

        # Vykreslenie histogramov
        for i in range(rozmer):
            for j in range(rozmer):
                vytvor_histogram(axs[i, j], *hodnoty[i*rozmer + j])

        # Nastavenie názvu a osí
        plt.tight_layout()
      #  plt.show()
        plt.savefig(f'Figure_{num}.png')

    except ValueError as ve:
        print(ve)
        continue
