# VisionPlay

Kamera ve el hareketleriyle oynanan mini oyunlar. Klavyeye veya fareye
dokunmadan, 2-3 metre uzaktan oynanacak sekilde tasarlandi.

Butun arayuz el ile kullaniliyor: bir butonun uzerine parmak ucunuzu getirip
**yumrugunuzu 2 saniye kapali tutunca** tiklaniyor.

## Modlar

| Mod | Nasil oynanir |
|---|---|
| **OKCULUK** | Bir el yayi tutar, diger el yumruk yapinca ok olusur; oku yaya yaklastirip geri cekin ve birakin. Hedef asagi yukari hareket eder. |
| **CIZIM** | Parmak ucunuzla havaya cizin. |
| **FRUIT NINJA** | Elinizdeki kilicla meyveleri kesin, bombalara dokunmayin. |
| **TOP SEKTIRME** | Topu elinizle havada tutun, yere dusurmeyin. |
| **YAKALA** | Iki elinizin arasindaki sepetle dusen cisimleri toplayin; siyah olanlar puan goturur. |
| **KOSTEBEK** | Alttan cikan kostebekleri elinizle geri sokun, kotu olanlara vurmayin. |

## Kurulum

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python setup_models.py
```

`setup_models.py`, MediaPipe'in el takibi modelini `models/` klasorune indirir.
Model dosyasi depoya dahil degildir.

## Calistirma

```bash
python main.py
```

Ilk calistirmada macOS kamera izni ister.

## macOS uygulamasi olarak paketleme

```bash
pip install pyinstaller
pyinstaller VisionPlay.spec --noconfirm
```

Sonuc: `dist/VisionPlay.app`

## Proje yapisi

- `core/` — kamera, el takibi, geometri, piksel yazi tipi, ortak arayuz parcalari
- `modes/` — her oyun modu ve menu ekranlari
- `assets/` — Aseprite ile cizilmis piksel gorseller ve yazi tipleri
- `main.py` — ekranlar arasi gecisi yoneten ana dongu

## Gereksinimler

Python 3.11+, bir web kamerasi ve iyi isik. macOS uzerinde gelistirildi.
